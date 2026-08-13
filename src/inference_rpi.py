#!/usr/bin/env python3
"""
Inferência em tempo real no Raspberry Pi 4 usando o modelo MobileNetV2
quantizado (INT8) em TensorFlow Lite.

Pipeline: captura frame da câmera -> pré-processa (resize 224x224 +
normalização) -> roda inferência -> aplica filtro de confiança (>=0.80) ->
imprime e publica o alerta estruturado via MQTT (src/mqtt_publisher.py).

Uso (no Raspberry Pi, com tflite-runtime instalado — ver
requirements-edge.txt):
    python src/inference_rpi.py --model models/mobilenetv2_int8_ptq.tflite --camera 0

Uso sem câmera física, para testar o pipeline com uma imagem estática:
    python src/inference_rpi.py --model models/mobilenetv2_int8_ptq.tflite --image caminho/para/folha.jpg
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.preprocessing import IMG_SIZE, resize_image

CONFIDENCE_THRESHOLD = 0.80  # mesmo limiar documentado no artigo N2, seção 3.1
DEFAULT_LABELS_PATH = Path(__file__).resolve().parent.parent / "models" / "class_names.txt"


def load_interpreter(model_path: str):
    """Carrega o interpretador TFLite. Prefere ``tflite_runtime`` (mais leve,
    recomendado no Raspberry Pi — ver requirements-edge.txt); usa
    ``tensorflow.lite`` como fallback em ambiente de desenvolvimento."""
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter

    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def load_labels(labels_path: Path) -> list[str]:
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Arquivo de labels não encontrado em {labels_path}. "
            "Ele é gerado por src/train.py junto com o modelo treinado."
        )
    return [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def preprocess_frame(frame: np.ndarray, input_dtype) -> np.ndarray:
    """Resize para 224x224 e normaliza conforme o tipo de entrada do
    modelo TFLite (int8 quantizado ou float32)."""
    resized = resize_image(frame, IMG_SIZE).astype(np.float32)

    if input_dtype == np.int8:
        # Modelo INT8: normaliza para [-1, 1] e depois quantiza para int8
        # (escala/zero-point simétricos, coerente com preprocess_input do MobileNetV2)
        normalized = (resized / 127.5) - 1.0
        quantized = np.clip(np.round(normalized / (1.0 / 128)), -128, 127).astype(np.int8)
        return np.expand_dims(quantized, axis=0)

    # Modelo FP32: normaliza para [-1, 1] (preprocess_input do MobileNetV2)
    normalized = (resized / 127.5) - 1.0
    return np.expand_dims(normalized, axis=0)


def run_inference(interpreter, input_tensor: np.ndarray) -> tuple[int, float]:
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    interpreter.set_tensor(input_details["index"], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    if output.dtype == np.int8:
        scale, zero_point = output_details["quantization"]
        output = (output.astype(np.float32) - zero_point) * scale

    class_id = int(np.argmax(output))
    confidence = float(output[class_id])
    return class_id, confidence


def capture_frame(camera_index: int):
    import cv2

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a câmera de índice {camera_index}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Falha ao capturar frame da câmera")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_image(image_path: str):
    from PIL import Image

    return np.array(Image.open(image_path).convert("RGB"))


def build_and_emit_alert(class_id: int, confidence: float, labels: list[str], args) -> None:
    class_name = labels[class_id] if class_id < len(labels) else f"classe_{class_id}"
    print(f"🔎 Predição: {class_name}  |  confiança: {confidence:.4f}")

    if confidence < CONFIDENCE_THRESHOLD:
        print(f"   (abaixo do limiar de {CONFIDENCE_THRESHOLD:.2f} — nenhum alerta enviado)")
        return

    print(f"🚨 ALERTA: {class_name} (confiança {confidence:.2%})")

    if not args.publish:
        return

    from mqtt_publisher import Alert, build_alert_payload, publish

    alert = Alert(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        lat=args.lat,
        lon=args.lon,
        timestamp=int(time.time()),
    )
    payload = build_alert_payload(alert)
    publish(args.broker, args.port, args.topic, payload)
    print(f"   publicado via MQTT ({len(payload)} bytes) em {args.broker}:{args.port}/{args.topic}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Caminho do modelo .tflite (ex.: models/mobilenetv2_int8_ptq.tflite)")
    parser.add_argument("--camera", type=int, default=None, help="Índice da câmera (ex.: 0 para /dev/video0)")
    parser.add_argument("--image", type=str, default=None, help="Caminho de uma imagem estática, alternativa à câmera")
    parser.add_argument("--labels", type=str, default=str(DEFAULT_LABELS_PATH), help="Arquivo texto com um nome de classe por linha")
    parser.add_argument("--publish", action="store_true", help="Publica o alerta via MQTT (requer --broker configurado)")
    parser.add_argument("--broker", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic", type=str, default="plantdisease/alerts")
    parser.add_argument("--lat", type=float, default=0.0, help="Latitude GPS do dispositivo")
    parser.add_argument("--lon", type=float, default=0.0, help="Longitude GPS do dispositivo")
    args = parser.parse_args()

    if args.camera is None and args.image is None:
        parser.error("informe --camera <índice> ou --image <caminho>")

    labels = load_labels(Path(args.labels))
    interpreter = load_interpreter(args.model)
    input_dtype = interpreter.get_input_details()[0]["dtype"]

    frame = capture_frame(args.camera) if args.camera is not None else load_image(args.image)

    input_tensor = preprocess_frame(frame, input_dtype)

    t0 = time.perf_counter()
    class_id, confidence = run_inference(interpreter, input_tensor)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"⏱️  Latência de inferência: {elapsed_ms:.1f} ms (medida neste dispositivo, agora)")
    build_and_emit_alert(class_id, confidence, labels, args)


if __name__ == "__main__":
    main()
