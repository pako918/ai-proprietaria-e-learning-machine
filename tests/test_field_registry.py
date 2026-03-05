"""Test suite per field_registry.py."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from field_registry import (
    registry, get_validator, FieldDef,
    _validate_cig, _validate_cup, _validate_cpv,
    _validate_importi, _validate_punteggi, _validate_date,
)


class TestValidators:
    def test_cig_valid(self):
        assert _validate_cig("ABC1234567") is True

    def test_cig_invalid(self):
        assert _validate_cig("short") is False
        assert _validate_cig("abc1234567") is False  # lowercase

    def test_cup_valid(self):
        assert _validate_cup("B12C34567890") is True

    def test_cup_invalid(self):
        assert _validate_cup("12345") is False

    def test_cpv_valid(self):
        assert _validate_cpv("71240000-2") is True

    def test_cpv_invalid(self):
        assert _validate_cpv("712400002") is False

    def test_importi_valid(self):
        assert _validate_importi("1.234.567,89") is True
        assert _validate_importi("100000") is True

    def test_importi_invalid(self):
        assert _validate_importi("0") is False
        assert _validate_importi("abc") is False

    def test_punteggi_valid(self):
        assert _validate_punteggi(80) is True
        assert _validate_punteggi(0) is True
        assert _validate_punteggi(100) is True

    def test_punteggi_invalid(self):
        assert _validate_punteggi(101) is False
        assert _validate_punteggi(-1) is False

    def test_date_valid(self):
        assert _validate_date("31/12/2025") is True
        assert _validate_date("01-06-25") is True

    def test_date_invalid(self):
        assert _validate_date("not a date") is False

    def test_get_validator(self):
        assert get_validator("cig") is not None
        assert get_validator("nonexistent") is None


class TestFieldRegistry:
    def test_get_all_returns_list(self):
        fields = registry.get_all()
        assert isinstance(fields, list)
        assert len(fields) > 20

    def test_get_known_field(self):
        fd = registry.get("cig")
        assert fd is not None
        assert fd.key == "cig"
        assert fd.label == "CIG"

    def test_get_unknown_field(self):
        assert registry.get("nonexistent_field_xyz") is None

    def test_get_patterns(self):
        patterns = registry.get_patterns()
        assert isinstance(patterns, dict)
        assert "cig" in patterns
        assert len(patterns["cig"]) > 0

    def test_get_keys(self):
        keys = registry.get_keys()
        assert "cig" in keys
        assert "importo_totale" in keys

    def test_get_by_category(self):
        cats = registry.get_by_category()
        assert "Identificativi" in cats
        assert "Importi" in cats

    def test_to_sections_json(self):
        sections = registry.to_sections_json()
        assert isinstance(sections, list)
        assert len(sections) > 0
        first = sections[0]
        assert "title" in first
        assert "fields" in first

    def test_custom_field_crud(self):
        key = "_test_custom_field"
        # Clean up if exists
        registry.delete_custom_field(key)

        # Create
        fd = registry.add_custom_field({
            "key": key,
            "label": "Test Campo",
            "category": "Test",
            "field_type": "text",
        })
        assert fd.key == key
        assert fd.is_custom is True

        # Read
        assert registry.get(key) is not None

        # Update
        updated = registry.update_custom_field(key, {"label": "Updated Label"})
        assert updated.label == "Updated Label"

        # Delete
        assert registry.delete_custom_field(key) is True
        assert registry.get(key) is None

    def test_add_custom_rejects_builtin_key(self):
        with pytest.raises(ValueError):
            registry.add_custom_field({"key": "cig", "label": "Duplicate CIG"})

    def test_add_custom_rejects_empty(self):
        with pytest.raises(ValueError):
            registry.add_custom_field({"key": "", "label": ""})
