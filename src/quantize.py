#!/usr/bin/env python3
"""
Versão em script (linha de comando/CI) da quantização feita no notebook
``notebooks/02_treinamento_quantizacao.ipynb``: converte o modelo FP32
treinado (src/train.py) para TFLite INT8 via Post-Training Quantization
(PTQ) e via Quantization-Aware Training (QAT), avalia AMBOS no conjunto de
teste e mede a latência local.

Correções deliberadas em relação ao notebook original (bugs reais
encontrados durante a auditoria do repositório):

1. A versão original do notebook **não avaliava de fato** o modelo QAT —
   usava ``acc_ptq + 0.003`` como valor "estimado" de acurácia QAT, sem
   rodar ``evaluate_tflite`` nele. Aqui o modelo QAT é avaliado da mesma
   forma que o PTQ, com números reais.
2. QAT direto (``tfmot.quantization.keras.quantize_model``) falhava com
   ``ValueError: Quantizing a keras Model inside another keras Model is
   not supported`` (a base MobileNetV2 estava aninhada como sub-modelo) e,
   após achatar o grafo, falhava de novo com ``Layer batch_normalization
   ... is not supported`` (a BatchNormalization da cabeça de classificação
   não está no padrão Conv2D+BN+ReLU que o esquema padrão de QAT
   reconhece). Corrigido com anotação seletiva: tudo é quantizado, exceto
   essa camada específica (``head_batchnorm``), que permanece em FP32
   dentro do modelo QAT.
3. Usa o pacote ``tf_keras`` diretamente (mesmo motivo do src/train.py):
   o tensorflow-model-optimization sempre usa tipos ``tf_keras``
   internamente, e misturar com ``tensorflow.keras``/``keras`` no mesmo
   grafo quebra o QAT silenciosamente.

A latência "no Raspberry Pi 4" é uma ESTIMATIVA: mede-se o tempo real
neste dispositivo (CPU de desenvolvimento) e aplica-se um fator de escala
da literatura (não é uma medição em hardware físico). Ver README.md,
seção "Limitações".

Uso:
    python src/quantize.py --data dataset/plantvillage_split --model models/mobilenetv2_fp32_best.h5 --output models/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

SEED = 42
BATCH_SIZE = 32

# Fatores de escala CPU-dev -> Raspberry Pi 4 (ARM Cortex-A72), baseados em
# literatura (não medidos em hardware físico neste projeto — ver README).
RPI4_SCALE_FP32 = 15.0
RPI4_SCALE_INT8 = 5.5


def representative_dataset_fn(data_dir: Path, img_size, num_samples: int = 200):
    import tf_keras
    from tf_keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(preprocessing_function=tf_keras.applications.mobilenet_v2.preprocess_input)
    gen = datagen.flow_from_directory(data_dir / "train", target_size=img_size, batch_size=1, class_mode=None, shuffle=True, seed=SEED)

    def gen_fn():
        for _ in range(num_samples):
            img = next(gen)
            yield [img.astype(np.float32)]

    return gen_fn


def convert_ptq(model, rep_dataset_fn):
    import tensorflow as tf

    converter_fp32 = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32 = converter_fp32.convert()

    converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = rep_dataset_fn
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.int8
    converter_int8.inference_output_type = tf.int8
    tflite_int8 = converter_int8.convert()

    return tflite_fp32, tflite_int8


def apply_qat(model, train_gen, val_gen, epochs: int = 3):
    import tensorflow_model_optimization as tfmot
    import tf_keras as keras
    from tensorflow_model_optimization.python.core.quantization.keras.quantize_annotate import (
        QuantizeAnnotate,
    )
    from tf_keras import callbacks

    def _skip_head_bn(layer):
        # A BatchNormalization da cabeça ("head_batchnorm", ver
        # src/train.py) não é suportada pelo esquema padrão de QAT nessa
        # posição — mantida em FP32, o resto do modelo é quantizado.
        # Usa QuantizeAnnotate(layer) diretamente (não a função pública
        # quantize_annotate_layer, que faz uma checagem de isinstance mais
        # estrita e falha para as camadas internas do MobileNetV2 nesta
        # combinação de versões de Keras/tf_keras).
        if layer.name == "head_batchnorm":
            return layer
        if isinstance(layer, QuantizeAnnotate):
            return layer
        return QuantizeAnnotate(layer)

    annotated_model = keras.models.clone_model(model, clone_function=_skip_head_bn)
    qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    qat_model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
    qat_model.fit(
        train_gen, epochs=epochs, validation_data=val_gen,
        callbacks=[callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)],
        verbose=2,
    )

    import tensorflow as tf
    converter_qat = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter_qat.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter_qat.convert()


def evaluate_tflite(tflite_bytes: bytes, generator, num_samples: int = 500):
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, f1_score

    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()[0]
    out_idx = interpreter.get_output_details()[0]["index"]
    in_scale, in_zero = in_details.get("quantization", (1.0, 0))

    y_true, y_pred = [], []
    generator.reset()
    seen = 0
    for batch_x, batch_y in generator:
        for j in range(len(batch_x)):
            if seen >= num_samples:
                break
            img = batch_x[j : j + 1].astype(np.float32)
            if in_details["dtype"] == np.int8:
                scale = in_scale if in_scale else 1.0
                img = (img / scale + in_zero).astype(np.int8)
            interpreter.set_tensor(in_details["index"], img)
            interpreter.invoke()
            out = interpreter.get_tensor(out_idx)
            y_pred.append(np.argmax(out))
            y_true.append(np.argmax(batch_y[j]))
            seen += 1
        if seen >= num_samples or seen >= generator.samples:
            break

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def benchmark_tflite(tflite_bytes: bytes, img_size, num_runs: int = 50):
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    in_idx = interpreter.get_input_details()[0]["index"]
    in_dtype = interpreter.get_input_details()[0]["dtype"]

    dummy = np.random.rand(1, *img_size, 3).astype(np.float32)
    if in_dtype == np.int8:
        dummy = dummy.astype(np.int8)

    for _ in range(5):
        interpreter.set_tensor(in_idx, dummy)
        interpreter.invoke()

    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(in_idx, dummy)
        interpreter.invoke()
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times)), float(np.std(times))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("dataset/plantvillage_split"))
    parser.add_argument("--model", type=Path, default=Path("models/mobilenetv2_fp32_best.h5"))
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument("--qat-epochs", type=int, default=3)
    args = parser.parse_args()

    import random

    import tensorflow as tf
    import tf_keras
    from tf_keras.preprocessing.image import ImageDataGenerator

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    IMG_SIZE = (224, 224)
    args.output.mkdir(parents=True, exist_ok=True)

    # Carregado via tf_keras (não tensorflow.keras/keras) — necessário para
    # que o QAT funcione (ver docstring do módulo).
    model = tf_keras.models.load_model(str(args.model))

    eval_datagen = ImageDataGenerator(preprocessing_function=tf_keras.applications.mobilenet_v2.preprocess_input)
    train_gen_qat = eval_datagen.flow_from_directory(args.data / "train", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=True, seed=SEED)
    val_gen = eval_datagen.flow_from_directory(args.data / "val", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)
    test_gen = eval_datagen.flow_from_directory(args.data / "test", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)

    print("Convertendo PTQ...")
    rep_fn = representative_dataset_fn(args.data, IMG_SIZE)
    tflite_fp32, tflite_ptq = convert_ptq(model, rep_fn)
    (args.output / "mobilenetv2_fp32.tflite").write_bytes(tflite_fp32)
    (args.output / "mobilenetv2_int8_ptq.tflite").write_bytes(tflite_ptq)

    print("Avaliando PTQ no conjunto de teste...")
    metrics_ptq = evaluate_tflite(tflite_ptq, test_gen)

    print("Aplicando QAT...")
    tflite_qat = apply_qat(model, train_gen_qat, val_gen, epochs=args.qat_epochs)
    (args.output / "mobilenetv2_int8_qat.tflite").write_bytes(tflite_qat)

    print("Avaliando QAT no conjunto de teste (medição real, sem fator fixo)...")
    metrics_qat = evaluate_tflite(tflite_qat, test_gen)

    print("Benchmark de latência local...")
    mean_fp32, _std_fp32 = benchmark_tflite(tflite_fp32, IMG_SIZE)
    mean_ptq, _std_ptq = benchmark_tflite(tflite_ptq, IMG_SIZE)
    mean_qat, _std_qat = benchmark_tflite(tflite_qat, IMG_SIZE)

    result = {
        "fp32": {
            "size_mb": len(tflite_fp32) / (1024 * 1024),
            "latency_local_ms": mean_fp32,
            "latency_rpi4_estimated_ms": mean_fp32 * RPI4_SCALE_FP32,
        },
        "int8_ptq": {
            **metrics_ptq,
            "size_mb": len(tflite_ptq) / (1024 * 1024),
            "latency_local_ms": mean_ptq,
            "latency_rpi4_estimated_ms": mean_ptq * RPI4_SCALE_INT8,
        },
        "int8_qat": {
            **metrics_qat,
            "size_mb": len(tflite_qat) / (1024 * 1024),
            "latency_local_ms": mean_qat,
            "latency_rpi4_estimated_ms": mean_qat * RPI4_SCALE_INT8,
        },
        "note": (
            "latency_rpi4_estimated_ms = latencia local * fator de escala da "
            "literatura (RPi4 Cortex-A72). NAO medido em hardware fisico."
        ),
    }
    (args.output / "quantize_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
