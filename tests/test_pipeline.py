"""Test suite per pipeline.py — estrazione regole e pipeline core."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from pipeline import compute_hash, RulesExtractor


class TestComputeHash:
    def test_bytes_input(self):
        h = compute_hash(b"test content")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256

    def test_string_input(self):
        h = compute_hash("test content")
        assert isinstance(h, str)
        assert len(h) == 64

    def test_deterministic(self):
        assert compute_hash(b"same") == compute_hash(b"same")

    def test_different_content(self):
        assert compute_hash(b"a") != compute_hash(b"b")


class TestRulesExtractor:
    @pytest.fixture
    def extractor(self):
        return RulesExtractor()

    def test_extract_cig(self, extractor):
        text = "Il CIG: ABC1234567 della presente gara."
        result, snippets, methods = extractor.extract(text)
        assert result.get("cig") == "ABC1234567"
        assert methods.get("cig") == "rules"

    def test_extract_cup(self, extractor):
        text = "CUP: B12C34567890"
        result, _, _ = extractor.extract(text)
        assert result.get("cup") == "B12C34567890"

    def test_extract_cpv(self, extractor):
        text = "Codice CPV: 71240000-2"
        result, _, _ = extractor.extract(text)
        assert result.get("cpv") == "71240000-2"

    def test_extract_importo(self, extractor):
        text = "L'importo complessivo dell'appalto è di € 1.234.567,89"
        result, _, _ = extractor.extract(text)
        assert result.get("importo_totale") is not None
        assert "1.234.567" in result["importo_totale"]

    def test_extract_punteggi(self, extractor):
        text = "offerta tecnica: 80 punti. Offerta economica: 20 punti."
        result, _, _ = extractor.extract(text)
        assert result.get("punteggio_tecnica") == 80
        assert result.get("punteggio_economica") == 20

    def test_extract_lotti(self, extractor):
        text = (
            "Gara suddivisa in 2 Lotto 1 servizi pulizia CIG: AAA1111111 "
            "importo base gara € 100.000,00 Lotto 2 servizi manutenzione "
            "CIG: BBB2222222 importo base gara € 200.000,00"
        )
        lotti = extractor.extract_lotti_detail(text)
        assert len(lotti) == 2
        assert lotti[0]["numero"] == 1
        assert lotti[1]["numero"] == 2

    def test_extract_categorie_ingegneria(self, extractor):
        text = "Categorie: E.22 S.03 IA.01"
        cats = extractor.extract_categorie_ingegneria(text)
        assert "E.22" in cats
        assert "S.03" in cats

    def test_detect_finanziamento_pnrr(self, extractor):
        text = "Il progetto è finanziato dal PNRR Missione 4"
        result = extractor.detect_finanziamento(text)
        assert result == "PNRR"

    def test_detect_finanziamento_none(self, extractor):
        text = "Nessun finanziamento specifico"
        result = extractor.detect_finanziamento(text)
        assert result is None

    def test_extract_boolean_flags(self, extractor):
        text = "È prevista la revisione dei prezzi. Conformità ai CAM obbligatoria."
        result, _, _ = extractor.extract(text)
        assert result.get("revisione_prezzi") is True
        assert result.get("conformita_cam") is True

    def test_extract_returns_three_dicts(self, extractor):
        result, snippets, methods = extractor.extract("Documento di test generico")
        assert isinstance(result, dict)
        assert isinstance(snippets, dict)
        assert isinstance(methods, dict)

    def test_extract_scadenze(self, extractor):
        text = "Scadenza presentazione offerte: 31/12/2025 ore 12:00"
        scadenze = extractor.extract_scadenze(text)
        assert "principale" in scadenze

    def test_extract_sopralluogo_obbligatorio(self, extractor):
        text = "Il sopralluogo obbligatorio deve essere effettuato presso le sedi."
        sop = extractor.extract_sopralluogo(text)
        assert sop["obbligatorio"] is True

    def test_extract_note_operative(self, extractor):
        text = (
            "Prevista inversione procedimentale. "
            "Soglia di sbarramento tecnica a 42 punti."
        )
        note = extractor.extract_note_operative(text)
        assert any("inversione" in n.lower() for n in note)
        assert any("sbarramento" in n.lower() for n in note)
