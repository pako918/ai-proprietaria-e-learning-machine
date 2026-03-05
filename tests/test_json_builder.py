"""Test suite per output_schema.py e json_builder.py"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from output_schema import AppaltoOutput, LottoImporto, DescrizioneLavori, StazioneAppaltante, RUP
from json_builder import build_output, build_output_with_methods


# ═════════════════════════════════════════════════════════════════════════════
# OUTPUT SCHEMA TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestAppaltoOutput:
    def test_minimal_creation(self):
        out = AppaltoOutput()
        assert out.cig is None
        assert out.note_particolari == []

    def test_full_creation(self):
        out = AppaltoOutput(
            cig={"lotto_1": "ABC1234567"},
            cup="J89B24000240004",
            oggetto_appalto="Test appalto",
            scadenza="01/01/2025",
        )
        assert out.cig["lotto_1"] == "ABC1234567"
        assert out.cup == "J89B24000240004"

    def test_model_dump_excludes_none(self):
        out = AppaltoOutput(cig={"lotto_1": "ABC1234567"})
        dumped = out.model_dump(exclude_none=True)
        assert "cup" not in dumped
        assert "cig" in dumped

    def test_lotto_importo(self):
        lotto = LottoImporto(lotto=1, importo_euro=100000.0)
        assert lotto.lotto == 1
        assert lotto.importo_euro == 100000.0

    def test_stazione_appaltante(self):
        sa = StazioneAppaltante(ente="Comune di Roma", rup=RUP(nome="Ing. Rossi"))
        assert sa.ente == "Comune di Roma"
        assert sa.rup.nome == "Ing. Rossi"


# ═════════════════════════════════════════════════════════════════════════════
# JSON BUILDER TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestBuildOutput:
    @pytest.fixture
    def nested_base(self):
        return {
            "informazioni_generali": {
                "titolo": "Affidamento progettazione",
                "CIG": "B80E0D5D7D",
                "CIG_per_lotto": [
                    {"lotto": 1, "CIG": "B80E0D5D7D"},
                    {"lotto": 2, "CIG": "B80E0D6E50"},
                ],
                "CUP": "J89B24000240004",
                "stazione_appaltante": {"denominazione": "Risorse per Roma S.p.A."},
                "RUP": {"nome": "Di Martino", "qualifica": "Arch."},
            },
            "tipo_procedura": {"tipo": "aperta", "criterio_aggiudicazione": "OEPV"},
            "suddivisione_lotti": {
                "numero_lotti": 2,
                "lotti": [
                    {"numero": 1, "importo_base_asta": 395700.0, "denominazione": "Lotto 1"},
                    {"numero": 2, "importo_base_asta": 249400.0, "denominazione": "Lotto 2"},
                ],
            },
            "importi_complessivi": {
                "importo_totale_gara": 645100.0,
                "quota_ribassabile_percentuale": 35,
            },
            "tempistiche": {"scadenza_offerte": "02/10/2025 ore 12:00"},
            "criteri_valutazione": {
                "offerta_tecnica": {"punteggio_massimo": 80, "criteri": []},
                "offerta_economica": {"punteggio_massimo": 20, "modalita_offerta": "ribasso_percentuale"},
            },
            "garanzie": {
                "garanzia_provvisoria": {"dovuta": False},
                "garanzia_definitiva": {"percentuale": 10},
            },
            "sopralluogo": {"obbligatorio": False},
            "requisiti_partecipazione": {
                "soggetti_ammessi": {},
                "idoneita_professionale": {},
                "capacita_economico_finanziaria": {},
                "capacita_tecnico_professionale": {},
                "gruppo_di_lavoro": {"figure_professionali": []},
            },
        }

    def test_cig_mapping(self, nested_base):
        result = build_output(nested_base)
        assert result["cig"] == {"lotto_1": "B80E0D5D7D", "lotto_2": "B80E0D6E50"}

    def test_cup_mapping(self, nested_base):
        result = build_output(nested_base)
        assert result["cup"] == "J89B24000240004"

    def test_oggetto_mapping(self, nested_base):
        result = build_output(nested_base)
        assert result["oggetto_appalto"] == "Affidamento progettazione"

    def test_stazione_appaltante_mapping(self, nested_base):
        result = build_output(nested_base)
        sa = result["stazione_appaltante"]
        assert sa["ente"] == "Risorse per Roma S.p.A."
        assert sa["rup"]["nome"] == "Arch. Di Martino"

    def test_lotti_mapping(self, nested_base):
        result = build_output(nested_base)
        lotti = result["descrizione_lavori_con_importo_totale"]["lotti"]
        assert len(lotti) == 2
        assert lotti[0]["lotto"] == 1
        assert lotti[0]["importo_euro"] == 395700.0
        assert lotti[0]["quota_fissa_65_percento_euro"] == 257205.0
        assert lotti[0]["quota_ribassabile_35_percento_euro"] == 138495.0

    def test_importo_totale_mapping(self, nested_base):
        result = build_output(nested_base)
        assert result["descrizione_lavori_con_importo_totale"]["importo_totale_procedura_euro"] == 645100.0

    def test_scadenza_mapping(self, nested_base):
        result = build_output(nested_base)
        assert result["scadenza"] == "02/10/2025 ore 12:00"

    def test_offerta_tecnica_mapping(self, nested_base):
        result = build_output(nested_base)
        assert result["offerta_tecnica"]["punteggio_massimo"] == 80.0

    def test_offerta_economica_mapping(self, nested_base):
        result = build_output(nested_base)
        assert result["offerta_economica"]["punteggio_massimo"] == 20.0
        assert "Ribasso percentuale" in result["offerta_economica"]["modalita"]

    def test_criteri_valutazione_mapping(self, nested_base):
        result = build_output(nested_base)
        crit = result["criteri_valutazione_offerta_tecnica"]
        assert crit["punteggio_totale"] == 100.0
        assert crit["ripartizione"]["offerta_tecnica"] == 80.0
        assert crit["ripartizione"]["offerta_economica"] == 20.0

    def test_garanzia_provvisoria_mapping(self, nested_base):
        result = build_output(nested_base)
        assert result["garanzia_provvisoria"]["richiesta"] is False
        assert result["garanzia_provvisoria"]["garanzia_definitiva_percentuale"] == 10.0

    def test_sopralluogo_mapping(self, nested_base):
        result = build_output(nested_base)
        assert result["sopralluogo"]["obbligatorio"] is False

    def test_vincoli_partecipazione_2_lotti(self, nested_base):
        result = build_output(nested_base)
        assert result["vincoli_partecipazione"]["vincolo_partecipazione_entrambi_lotti"] is True

    def test_note_particolari_generated(self, nested_base):
        result = build_output(nested_base)
        assert any("2 lotti" in n for n in result["note_particolari"])

    def test_empty_input(self):
        result = build_output({})
        assert isinstance(result, dict)
        assert "note_particolari" in result

    def test_single_cig_fallback(self):
        nested = {"informazioni_generali": {"CIG": "ABCDEF1234"}}
        result = build_output(nested)
        assert result["cig"] == {"lotto_1": "ABCDEF1234"}


class TestBuildOutputWithMethods:
    def test_returns_tuple(self):
        nested = {"informazioni_generali": {"CIG": "ABC1234567"}}
        output, methods = build_output_with_methods(nested)
        assert isinstance(output, dict)
        assert isinstance(methods, dict)

    def test_methods_populated(self):
        nested = {
            "informazioni_generali": {"CIG": "ABC1234567", "CUP": "J89B24000240004"},
        }
        output, methods = build_output_with_methods(nested)
        assert "cig" in methods
        assert methods["cig"] == "rules"

    def test_empty_methods(self):
        output, methods = build_output_with_methods({})
        assert isinstance(methods, dict)
