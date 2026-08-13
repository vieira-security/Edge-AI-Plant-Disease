#!/usr/bin/env python3
"""
Versão em script (linha de comando/CI) do treinamento feito no notebook
``notebooks/02_treinamento_quantizacao.ipynb``: fine-tuning em duas fases
de um MobileNetV2 pré-treinado no ImageNet sobre o dataset PlantVillage.

Diferenças deliberadas em relação ao notebook original (bugs reais
encontrados e corrigidos durante a auditoria do repositório):

1. Usa uma divisão treino/validação/teste com **três pastas físicas
   separadas** (``dataset/plantvillage_split/{train,val,test}``) em vez de
   reaproveitar o split de validação como "teste" (o notebook original
   usava a mesma partição para os dois, o que inflava a métrica de teste
   por vazamento de dados).
2. Constrói o modelo com ``base.input``/``base.output`` diretamente (grafo
   funcional único), em vez de chamar ``base(inputs)`` como sub-modelo
   aninhado — essa segunda forma quebra o QAT (ver src/quantize.py) com
   ``ValueError: Quantizing a keras Model inside another keras Model is
   not supported``.
3. Usa o pacote ``tf_keras`` (Keras 2 legado) diretamente, em vez de
   ``tensorflow.keras``, para que o modelo salvo seja diretamente
   compatível com o QAT feito em src/quantize.py (tensorflow-model-
   optimization exige tipos ``tf_keras``; misturar com
   ``tensorflow.keras``/``keras`` no mesmo grafo quebra silenciosamente).

Uso:
    python src/train.py --data dataset/plantvillage_split --epochs-head 10 \\
        --epochs-fine 40 --output models/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.preprocessing import IMG_SIZE

SEED = 42
BATCH_SIZE = 32
BASE_LR = 1e-3
FINE_LR = 1e-4
UNFREEZE_FROM = 100


def build_model(num_classes: int, trainable_base: bool = False):
    """Constrói MobileNetV2 com cabeça de classificação customizada."""
    import tf_keras as keras
    from tf_keras import layers
    from tf_keras.applications import MobileNetV2

    base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = trainable_base

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization(name="head_batchnorm")(x)  # nomeada p/ QAT seletivo (ver src/quantize.py)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(base.input, outputs), base


def build_generators(data_dir: Path):
    import tf_keras
    from tf_keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(
        preprocessing_function=tf_keras.applications.mobilenet_v2.preprocess_input,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.20,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.75, 1.25],
        fill_mode="nearest",
    )
    eval_datagen = ImageDataGenerator(preprocessing_function=tf_keras.applications.mobilenet_v2.preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        data_dir / "train", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=True, seed=SEED,
    )
    val_gen = eval_datagen.flow_from_directory(
        data_dir / "val", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False,
    )
    test_gen = eval_datagen.flow_from_directory(
        data_dir / "test", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False,
    )
    return train_gen, val_gen, test_gen


def evaluate(model, generator):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    generator.reset()
    y_pred_proba = model.predict(generator, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = generator.classes[: len(y_pred)]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("dataset/plantvillage_split"))
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument("--epochs-head", type=int, default=10)
    parser.add_argument("--epochs-fine", type=int, default=40)
    args = parser.parse_args()

    import random

    import tensorflow as tf
    import tf_keras as keras
    from tf_keras import callbacks

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    args.output.mkdir(parents=True, exist_ok=True)

    train_gen, val_gen, test_gen = build_generators(args.data)
    class_names = list(train_gen.class_indices.keys())
    (args.output / "class_names.txt").write_text("\n".join(class_names), encoding="utf-8")

    print(f"Classes: {len(class_names)} | treino: {train_gen.samples} | val: {val_gen.samples} | teste: {test_gen.samples}")

    # Fase 1: cabeça
    model, base_model = build_model(num_classes=len(class_names), trainable_base=False)
    model.compile(optimizer=keras.optimizers.Adam(BASE_LR), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        train_gen, epochs=args.epochs_head, validation_data=val_gen,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ],
        verbose=2,
    )

    # Fase 2: fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:UNFREEZE_FROM]:
        layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(FINE_LR), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        train_gen, epochs=args.epochs_fine, validation_data=val_gen,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7),
        ],
        verbose=2,
    )

    model.save(str(args.output / "mobilenetv2_fp32_best.h5"))

    metrics = evaluate(model, test_gen)
    print("Métricas no conjunto de TESTE (pasta separada, sem vazamento):", metrics)

    (args.output / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
