"""Test suite per i moduli di utilità (utils.py)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from utils import (
    clean_string, normalize_amount, parse_number_word,
    find_value_context, first_match, all_matches, extract_int,
)


# ── clean_string ──────────────────────────────────────────────────

class TestCleanString:
    def test_collapses_whitespace(self):
        assert clean_string("ciao   mondo") == "ciao mondo"

    def test_strips_trailing_punctuation(self):
        assert clean_string("valore;") == "valore"
        assert clean_string("valore,") == "valore"
        assert clean_string("valore:  ") == "valore"

    def test_none_passthrough(self):
        assert clean_string(None) is None

    def test_empty_string(self):
        assert clean_string("") == ""

    def test_multiline(self):
        assert clean_string("  riga 1  \n  riga 2  ") == "riga 1 riga 2"


# ── normalize_amount ──────────────────────────────────────────────

class TestNormalizeAmount:
    def test_italian_format(self):
        result = normalize_amount("1.234,56")
        assert result == "€ 1.234,56"

    def test_with_euro_sign(self):
        result = normalize_amount("€ 100.000,00")
        assert result == "€ 100.000,00"

    def test_english_format(self):
        result = normalize_amount("1,234.56")
        assert result == "€ 1.234,56"

    def test_none(self):
        assert normalize_amount(None) is None

    def test_empty(self):
        assert normalize_amount("") is None

    def test_simple_number(self):
        result = normalize_amount("500")
        assert "500" in result


# ── parse_number_word ─────────────────────────────────────────────

class TestParseNumberWord:
    def test_word_due(self):
        assert parse_number_word("due") == 2

    def test_word_cinque(self):
        assert parse_number_word("cinque") == 5

    def test_numeric(self):
        assert parse_number_word("42") == 42

    def test_none(self):
        assert parse_number_word(None) is None

    def test_empty(self):
        assert parse_number_word("") is None

    def test_with_extra_chars(self):
        assert parse_number_word("3 lotti") == 3


# ── find_value_context ────────────────────────────────────────────

class TestFindValueContext:
    SAMPLE_TEXT = (
        "La stazione appaltante è il Comune di Roma. "
        "L'importo totale dell'appalto è di € 1.234.567,89. "
        "Il CIG è ABC1234567. La scadenza è il 31/12/2025."
    )

    def test_direct_match(self):
        ctx = find_value_context(self.SAMPLE_TEXT, "ABC1234567")
        assert "ABC1234567" in ctx

    def test_word_fallback(self):
        ctx = find_value_context(self.SAMPLE_TEXT, "Comune di Roma")
        assert "Comune" in ctx

    def test_number_fallback(self):
        ctx = find_value_context(self.SAMPLE_TEXT, "€ 1.234.567,89")
        assert "1234567" in ctx or "importo" in ctx.lower()

    def test_empty_value(self):
        assert find_value_context(self.SAMPLE_TEXT, "") == ""

    def test_empty_text(self):
        assert find_value_context("", "test") == ""

    def test_not_found(self):
        assert find_value_context(self.SAMPLE_TEXT, "ZZZZZZZZZ") == ""


# ── first_match ───────────────────────────────────────────────────

class TestFirstMatch:
    def test_basic_match(self):
        result = first_match("CIG: ABC1234567", [r'CIG[\s:]*([A-Z0-9]{10})'])
        assert result == "ABC1234567"

    def test_no_match(self):
        result = first_match("nessun match qui", [r'CIG[\s:]*([A-Z0-9]{10})'])
        assert result is None

    def test_multiple_patterns(self):
        result = first_match("importo: 100.000", [
            r'totale[\s:]*(\d+)',
            r'importo[\s:]*([0-9.,]+)',
        ])
        assert result == "100.000"

    def test_empty_patterns(self):
        assert first_match("testo", []) is None

    def test_invalid_regex(self):
        assert first_match("testo", [r'[invalid']) is None


# ── all_matches ───────────────────────────────────────────────────

class TestAllMatches:
    def test_multiple_results(self):
        text = "CIG: AAA1111111 e CIG: BBB2222222"
        result = all_matches(text, [r'CIG[\s:]*([A-Z0-9]{10})'])
        assert len(result) == 2
        assert "AAA1111111" in result
        assert "BBB2222222" in result

    def test_dedup(self):
        text = "CIG AAA1111111 - ripeto CIG AAA1111111"
        result = all_matches(text, [r'CIG\s+([A-Z0-9]{10})'])
        assert len(result) == 1


# ── extract_int ───────────────────────────────────────────────────

class TestExtractInt:
    def test_extracts_number(self):
        result = extract_int("punteggio tecnico: 80 punti", [r'tecnico[\s:]*(\d+)'])
        assert result == 80

    def test_no_match(self):
        assert extract_int("nessun numero", [r'valore[\s:]*(\d+)']) is None
