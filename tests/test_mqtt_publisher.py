"""Testes para src/mqtt_publisher.py — montagem e parsing do payload de alerta."""
import time

import pytest

from src.mqtt_publisher import (
    PAYLOAD_SIZE_BYTES,
    Alert,
    build_alert_payload,
    build_json_payload,
    decode_alert_payload,
    measure_payload_sizes,
)


def _sample_alert(**overrides):
    defaults = {
        "class_id": 23,
        "class_name": "Tomato___Late_blight",
        "confidence": 0.91,
        "lat": -23.5505,
        "lon": -46.6333,
        "timestamp": 1_780_000_000,
    }
    defaults.update(overrides)
    return Alert(**defaults)


class TestBuildAlertPayload:
    def test_payload_size_is_14_bytes(self):
        payload = build_alert_payload(_sample_alert())
        assert len(payload) == PAYLOAD_SIZE_BYTES == 14

    def test_payload_size_within_article_range(self):
        # Artigo N2, seção 3.1: payload compacto de 10-30 bytes
        payload = build_alert_payload(_sample_alert())
        assert 10 <= len(payload) <= 30

    def test_rejects_class_id_out_of_range(self):
        with pytest.raises(ValueError):
            build_alert_payload(_sample_alert(class_id=300))

    def test_rejects_confidence_out_of_range(self):
        with pytest.raises(ValueError):
            build_alert_payload(_sample_alert(confidence=1.5))


class TestDecodeAlertPayload:
    def test_roundtrip_preserves_class_and_timestamp(self):
        alert = _sample_alert()
        payload = build_alert_payload(alert)
        decoded = decode_alert_payload(payload)
        assert decoded["class_id"] == alert.class_id
        assert decoded["timestamp"] == alert.timestamp

    def test_roundtrip_preserves_confidence_within_quantization_error(self):
        alert = _sample_alert(confidence=0.91)
        payload = build_alert_payload(alert)
        decoded = decode_alert_payload(payload)
        # confiança é quantizada em 1 byte (0-255), então há pequeno erro de arredondamento
        assert decoded["confidence"] == pytest.approx(0.91, abs=1 / 255)

    def test_roundtrip_preserves_gps_within_float32_precision(self):
        alert = _sample_alert(lat=-23.5505, lon=-46.6333)
        payload = build_alert_payload(alert)
        decoded = decode_alert_payload(payload)
        assert decoded["lat"] == pytest.approx(-23.5505, abs=1e-3)
        assert decoded["lon"] == pytest.approx(-46.6333, abs=1e-3)

    def test_rejects_wrong_size_payload(self):
        with pytest.raises(ValueError):
            decode_alert_payload(b"\x00\x01\x02")


class TestMeasurePayloadSizes:
    def test_binary_is_smaller_than_json(self):
        sizes = measure_payload_sizes(_sample_alert())
        assert sizes["binary_bytes"] < sizes["json_bytes"]

    def test_reduction_percentage_is_positive(self):
        sizes = measure_payload_sizes(_sample_alert())
        assert sizes["reduction_vs_json_pct"] > 0

    def test_json_payload_is_valid_and_contains_class_name(self):
        import json

        alert = _sample_alert()
        raw = build_json_payload(alert)
        obj = json.loads(raw)
        assert obj["class_name"] == alert.class_name
        assert obj["class_id"] == alert.class_id


class TestAlertDefaults:
    def test_timestamp_close_to_now_when_freshly_built(self):
        alert = _sample_alert(timestamp=int(time.time()))
        payload = build_alert_payload(alert)
        decoded = decode_alert_payload(payload)
        assert abs(decoded["timestamp"] - time.time()) < 5
