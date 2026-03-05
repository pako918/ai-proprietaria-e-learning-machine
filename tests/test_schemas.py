"""Test suite per schemas.py — validazione Pydantic."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from schemas import full_validation


class TestFullValidation:
    def test_valid_basic_result(self):
        data = {
            "cig": "ABC1234567",
            "importo_totale": "€ 100.000,00",
            "punteggio_tecnica": 80,
            "punteggio_economica": 20,
        }
        result = full_validation(data)
        assert isinstance(result, dict)
        assert "coherence" in result or "warnings" in result

    def test_empty_input(self):
        result = full_validation({})
        assert isinstance(result, dict)

    def test_with_metadata_fields(self):
        data = {
            "cig": "ABC1234567",
            "_doc_id": "test123",
            "_confidence": 50.0,
        }
        result = full_validation(data)
        assert isinstance(result, dict)
