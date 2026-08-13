#!/usr/bin/env python3
"""
Monta o payload compacto de alerta (classe, confiança, GPS, timestamp) e
publica via MQTT — ver README.md, seção "Como Reproduzir" /
"Implantar no Raspberry Pi 4".

O payload binário compacto dá base empírica ao número de redução de
tráfego de rede citado no artigo (Tabela 3 do relatório N2): antes desse
script, a estimativa era só teórica (30 bytes "no papel"); aqui o tamanho
real em bytes é medido com ``len(payload)`` sobre dados de verdade, e
comparado ao equivalente em JSON (formato "legível" que um sistema ingênuo
mandaria em vez do payload binário).

Formato do payload binário (struct, little-endian):
    B   class_id     (1 byte,  0-255)
    B   confidence   (1 byte,  confiança * 255, arredondada)
    f   latitude     (4 bytes, float32)
    f   longitude    (4 bytes, float32)
    I   timestamp    (4 bytes, unix epoch, uint32)
Total: 14 bytes (dentro da faixa de 10-30 bytes citada no artigo N2, seção 3.1).

Uso:
    python src/mqtt_publisher.py --class-id 23 --class-name Tomato___Late_blight \\
        --confidence 0.91 --lat -23.5505 --lon -46.6333 \\
        --broker localhost --topic plantdisease/alerts
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from dataclasses import dataclass

PAYLOAD_FORMAT = "<BBffI"  # little-endian: class_id, confidence, lat, lon, timestamp
PAYLOAD_SIZE_BYTES = struct.calcsize(PAYLOAD_FORMAT)  # 14 bytes

CONFIDENCE_THRESHOLD = 0.80  # mesmo limiar usado em src/inference_rpi.py


@dataclass
class Alert:
    class_id: int
    class_name: str
    confidence: float
    lat: float
    lon: float
    timestamp: int


def build_alert_payload(alert: Alert) -> bytes:
    """Serializa um alerta no formato binário compacto (14 bytes)."""
    if not (0 <= alert.class_id <= 255):
        raise ValueError("class_id deve estar entre 0 e 255 (1 byte)")
    if not (0.0 <= alert.confidence <= 1.0):
        raise ValueError("confidence deve estar entre 0.0 e 1.0")

    confidence_byte = round(alert.confidence * 255)
    return struct.pack(
        PAYLOAD_FORMAT,
        alert.class_id,
        confidence_byte,
        alert.lat,
        alert.lon,
        alert.timestamp,
    )


def decode_alert_payload(payload: bytes) -> dict:
    """Decodifica um payload binário de volta para um dicionário."""
    if len(payload) != PAYLOAD_SIZE_BYTES:
        raise ValueError(
            f"Payload de tamanho inesperado: {len(payload)} bytes "
            f"(esperado {PAYLOAD_SIZE_BYTES})"
        )
    class_id, confidence_byte, lat, lon, timestamp = struct.unpack(PAYLOAD_FORMAT, payload)
    return {
        "class_id": class_id,
        "confidence": round(confidence_byte / 255, 4),
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "timestamp": timestamp,
    }


def build_json_payload(alert: Alert) -> bytes:
    """Serializa o mesmo alerta em JSON legível — usado apenas como
    referência de comparação para medir a economia de bytes do formato
    binário compacto (ver measure_payload_sizes)."""
    obj = {
        "class_id": alert.class_id,
        "class_name": alert.class_name,
        "confidence": round(alert.confidence, 4),
        "lat": alert.lat,
        "lon": alert.lon,
        "timestamp": alert.timestamp,
    }
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def measure_payload_sizes(alert: Alert) -> dict:
    """Mede em bytes o payload binário compacto vs. o equivalente JSON,
    dando base empírica (e não só teórica) à redução de payload."""
    binary_payload = build_alert_payload(alert)
    json_payload = build_json_payload(alert)
    reduction_pct = (1 - len(binary_payload) / len(json_payload)) * 100
    return {
        "binary_bytes": len(binary_payload),
        "json_bytes": len(json_payload),
        "reduction_vs_json_pct": round(reduction_pct, 1),
    }


def publish(broker: str, port: int, topic: str, payload: bytes, qos: int = 1) -> None:
    """Publica o payload via MQTT usando paho-mqtt."""
    import paho.mqtt.publish as mqtt_publish

    mqtt_publish.single(topic, payload=payload, qos=qos, hostname=broker, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--class-id", type=int, required=True, help="Índice numérico da classe (0-37)")
    parser.add_argument("--class-name", type=str, required=True, help="Nome da classe (ex.: Tomato___Late_blight)")
    parser.add_argument("--confidence", type=float, required=True, help="Confiança da predição (0.0-1.0)")
    parser.add_argument("--lat", type=float, required=True, help="Latitude GPS")
    parser.add_argument("--lon", type=float, required=True, help="Longitude GPS")
    parser.add_argument("--broker", type=str, default="localhost", help="Endereço do broker MQTT")
    parser.add_argument("--port", type=int, default=1883, help="Porta do broker MQTT")
    parser.add_argument("--topic", type=str, default="plantdisease/alerts", help="Tópico MQTT")
    parser.add_argument("--dry-run", action="store_true", help="Não publica, apenas mostra o payload e seu tamanho")
    args = parser.parse_args()

    if args.confidence < CONFIDENCE_THRESHOLD:
        print(
            f"⚠️  Confiança {args.confidence:.2f} abaixo do limiar de "
            f"{CONFIDENCE_THRESHOLD:.2f} — alerta não deveria ser enviado."
        )
        return

    alert = Alert(
        class_id=args.class_id,
        class_name=args.class_name,
        confidence=args.confidence,
        lat=args.lat,
        lon=args.lon,
        timestamp=int(time.time()),
    )

    payload = build_alert_payload(alert)
    sizes = measure_payload_sizes(alert)

    print(f"Payload binário: {payload.hex()} ({sizes['binary_bytes']} bytes)")
    print(f"Equivalente JSON: {sizes['json_bytes']} bytes")
    print(f"Redução vs. JSON: {sizes['reduction_vs_json_pct']}%")

    if args.dry_run:
        print("(--dry-run) Nada foi publicado.")
        return

    publish(args.broker, args.port, args.topic, payload)
    print(f"✅ Publicado em {args.broker}:{args.port}/{args.topic}")


if __name__ == "__main__":
    main()
