"""Testes para src/utils/preprocessing.py"""
import numpy as np
import pytest

from src.utils.preprocessing import (
    IMG_SIZE,
    SHARPNESS_THRESHOLD,
    is_low_quality,
    laplacian_variance,
    normalize_mobilenet,
    normalize_unit_scale,
    parse_class_folder_name,
    preprocess_for_inference,
    resize_image,
)


def _random_image(h=300, w=300, c=3):
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)


class TestResizeImage:
    def test_resizes_to_target_size(self):
        img = _random_image(100, 150)
        out = resize_image(img, size=(224, 224))
        assert out.shape[:2] == (224, 224)

    def test_default_size_is_224(self):
        assert IMG_SIZE == (224, 224)


class TestNormalizeMobilenet:
    def test_output_range_is_minus1_to_1(self):
        img = np.array([[0, 127.5, 255]], dtype=np.uint8)
        out = normalize_mobilenet(img)
        assert out.min() >= -1.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_black_pixel_maps_to_minus_one(self):
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        out = normalize_mobilenet(img)
        assert np.allclose(out, -1.0)

    def test_white_pixel_maps_to_approximately_one(self):
        img = np.full((2, 2, 3), 255, dtype=np.uint8)
        out = normalize_mobilenet(img)
        assert np.allclose(out, 1.0, atol=1e-2)


class TestNormalizeUnitScale:
    def test_output_range_is_0_to_1(self):
        img = np.array([[0, 128, 255]], dtype=np.uint8)
        out = normalize_unit_scale(img)
        assert out.min() == pytest.approx(0.0)
        assert out.max() == pytest.approx(1.0)

    def test_dtype_is_float32(self):
        img = _random_image()
        out = normalize_unit_scale(img)
        assert out.dtype == np.float32


class TestPreprocessForInference:
    def test_adds_batch_dimension(self):
        img = _random_image(100, 100)
        out = preprocess_for_inference(img)
        assert out.shape == (1, 224, 224, 3)

    def test_output_is_normalized(self):
        img = _random_image(100, 100)
        out = preprocess_for_inference(img)
        assert out.min() >= -1.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6


class TestLaplacianVariance:
    def test_uniform_image_has_near_zero_variance(self):
        flat = np.full((100, 100, 3), 128, dtype=np.uint8)
        assert laplacian_variance(flat) < 1.0

    def test_noisy_image_has_high_variance(self):
        rng = np.random.default_rng(0)
        noisy = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
        assert laplacian_variance(noisy) > SHARPNESS_THRESHOLD


class TestIsLowQuality:
    def test_uniform_image_is_low_quality(self):
        flat = np.full((100, 100, 3), 128, dtype=np.uint8)
        assert is_low_quality(flat) is True

    def test_noisy_image_is_not_low_quality(self):
        rng = np.random.default_rng(0)
        noisy = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
        assert is_low_quality(noisy) is False

    def test_custom_threshold(self):
        flat = np.full((100, 100, 3), 128, dtype=np.uint8)
        assert is_low_quality(flat, threshold=-1.0) is False


class TestParseClassFolderName:
    def test_parses_species_and_condition(self):
        species, condition, is_healthy = parse_class_folder_name("Tomato___Late_blight")
        assert species == "Tomato"
        assert condition == "Late blight"
        assert is_healthy is False

    def test_detects_healthy_class(self):
        species, _condition, is_healthy = parse_class_folder_name("Apple___healthy")
        assert species == "Apple"
        assert is_healthy is True

    def test_handles_missing_separator(self):
        species, condition, is_healthy = parse_class_folder_name("UnknownFolder")
        assert species == "UnknownFolder"
        assert condition == "Unknown"
        assert is_healthy is False

    def test_handles_multi_word_species(self):
        species, _, _ = parse_class_folder_name("Corn_(maize)___Common_rust_")
        assert species == "Corn (maize)"
