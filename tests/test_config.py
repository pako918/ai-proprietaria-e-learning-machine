"""Test suite per config.py."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from config import (
    BASE_DIR, DATA_DIR, MODEL_DIR, UPLOAD_DIR, DB_PATH,
    SERVER_HOST, SERVER_PORT, CORS_ORIGINS,
    MAX_UPLOAD_SIZE_BYTES, MIN_TEXT_LENGTH,
    MIN_SAMPLES_TRAIN, MIN_SAMPLES_CV, DEFAULT_CONFIDENCE_THRESHOLD,
    DATA_QUALITY_WEIGHTS, API_VERSION,
)


class TestPaths:
    def test_base_dir_exists(self):
        assert BASE_DIR.exists()

    def test_data_dir_exists(self):
        assert DATA_DIR.exists()

    def test_model_dir_exists(self):
        assert MODEL_DIR.exists()

    def test_upload_dir_exists(self):
        assert UPLOAD_DIR.exists()

    def test_db_path_under_data(self):
        assert str(DB_PATH).startswith(str(DATA_DIR))


class TestServerConfig:
    def test_host_is_string(self):
        assert isinstance(SERVER_HOST, str)

    def test_port_is_int(self):
        assert isinstance(SERVER_PORT, int)
        assert 1 <= SERVER_PORT <= 65535

    def test_cors_is_list(self):
        assert isinstance(CORS_ORIGINS, list)
        assert len(CORS_ORIGINS) > 0


class TestUploadConfig:
    def test_max_size_positive(self):
        assert MAX_UPLOAD_SIZE_BYTES > 0

    def test_min_text_length_positive(self):
        assert MIN_TEXT_LENGTH > 0


class TestMLConfig:
    def test_min_samples_positive(self):
        assert MIN_SAMPLES_TRAIN >= 1
        assert MIN_SAMPLES_CV > MIN_SAMPLES_TRAIN

    def test_confidence_threshold_range(self):
        assert 0 < DEFAULT_CONFIDENCE_THRESHOLD < 1

    def test_data_quality_weights(self):
        assert "correction" in DATA_QUALITY_WEIGHTS
        assert DATA_QUALITY_WEIGHTS["correction"] == 1.0
        assert all(0 <= v <= 1 for v in DATA_QUALITY_WEIGHTS.values())

    def test_api_version(self):
        assert isinstance(API_VERSION, str)
        assert len(API_VERSION) > 0
