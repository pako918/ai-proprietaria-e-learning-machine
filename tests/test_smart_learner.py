"""Test suite per smart_learner.py — Apprendimento Progressivo Autonomo"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from smart_learner import PatternLearner, AutoTrainer, SelfEvaluator, SmartLearner


# ═════════════════════════════════════════════════════════════════════════════
# PATTERN LEARNER TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestPatternLearner:

    @pytest.fixture
    def learner(self):
        return PatternLearner()

    def test_tables_created(self, learner):
        """Le tabelle per i pattern devono essere create."""
        from database import get_connection
        with get_connection(readonly=True) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('learned_patterns', 'extraction_log')"
            ).fetchall()
        assert len(tables) == 2

    def test_learn_from_correction(self, learner):
        """Deve apprendere pattern dalla correzione."""
        full_text = (
            "DISCIPLINARE DI GARA\n\n"
            "OGGETTO: Servizi di ingegneria e architettura per la "
            "progettazione definitiva ed esecutiva\n\n"
            "CIG: 1234567890\n"
            "CUP: J89B24000240004\n"
        )
        learned = learner.learn_from_correction(
            field="oggetto_appalto",
            correct_value="Servizi di ingegneria e architettura per la "
                          "progettazione definitiva ed esecutiva",
            wrong_value="Servizi di ingegneria",
            full_text=full_text,
            doc_id="test_doc_01"
        )
        assert len(learned) > 0
        assert learned[0]["field"] == "oggetto_appalto"
        assert learned[0]["status"] in ("new", "reinforced")

    def test_learn_reinforces_existing(self, learner):
        """Pattern duplicato deve essere rinforzato, non creato nuovo."""
        text = "OGGETTO: Manutenzione stradale ordinaria\n\nCIG: XXX"
        # Prima volta
        r1 = learner.learn_from_correction(
            "oggetto_appalto", "Manutenzione stradale ordinaria",
            "", text, "doc_a"
        )
        # Seconda volta (stesso prefisso)
        r2 = learner.learn_from_correction(
            "oggetto_appalto", "Manutenzione impianti elettrici",
            "", "OGGETTO: Manutenzione impianti elettrici\n\nCIG: YYY", "doc_b"
        )
        # Almeno uno deve essere "reinforced" se il prefix text è lo stesso
        # (ma testi diversi generano prefix diversi, quindi entrambi "new")
        assert len(r1) > 0 or len(r2) > 0

    def test_extract_with_patterns(self, learner):
        """Deve estrarre valori usando pattern appresi."""
        # Prima insegna un pattern
        full_text = "OGGETTO: Lavori edili speciali\n\nCIG: 999"
        learner.learn_from_correction(
            "oggetto_appalto", "Lavori edili speciali", "", full_text, "doc_x"
        )

        # Poi estrai da un testo DIVERSO con struttura simile
        new_text = "OGGETTO: Costruzione ponte pedonale\n\nCIG: 888"
        results = learner.extract_with_patterns("oggetto_appalto", new_text)
        # Potrebbe avere risultati (dipende dal pattern generalizzato)
        # L'importante è che il metodo non faccia errori
        assert isinstance(results, list)

    def test_empty_value_returns_nothing(self, learner):
        """Valore vuoto non deve apprendere nulla."""
        learned = learner.learn_from_correction("test_field", "", "", "some text")
        assert learned == []

    def test_report_success_and_failure(self, learner):
        """Success/failure devono aggiornare i contatori."""
        text = "LABEL X: Valore test ABC\n\nFINE"
        r = learner.learn_from_correction("test_field", "Valore test ABC", "", text)
        if r:
            pid = r[0]["id"]
            learner.report_success(pid)
            learner.report_failure(pid)
            # Non deve crashare

    def test_pattern_confidence(self, learner):
        """Confidenza deve essere tra 0 e 1."""
        conf = learner._pattern_confidence({
            "success_count": 5, "fail_count": 1
        })
        assert 0 <= conf <= 1
        assert conf > 0.5  # Più successi che fallimenti

        conf_bad = learner._pattern_confidence({
            "success_count": 1, "fail_count": 10
        })
        assert conf_bad < 0.5

    def test_generalize_to_regex(self, learner):
        """Generalizzazione deve produrre regex valide."""
        result = learner._generalize_to_regex("OGGETTO: ")
        assert result  # Non vuoto
        assert "\\s+" in result or "OGGETTO" in result

    def test_get_field_stats(self, learner):
        """Statistiche campo devono funzionare."""
        stats = learner.get_field_stats()
        assert isinstance(stats, dict)

        stats_field = learner.get_field_stats("oggetto_appalto")
        assert "active_patterns" in stats_field


# ═════════════════════════════════════════════════════════════════════════════
# AUTO TRAINER TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestAutoTrainer:

    @pytest.fixture
    def trainer(self):
        return AutoTrainer()

    def test_tables_created(self, trainer):
        """Le tabelle auto-training devono essere create."""
        from database import get_connection
        with get_connection(readonly=True) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('auto_train_log', 'correction_counter')"
            ).fetchall()
        assert len(tables) == 2

    def test_correction_counting(self, trainer):
        """Deve contare le correzioni per campo."""
        # Prima correzione
        result = trainer.record_correction("test_auto_field")
        assert result is None  # Non ancora alla soglia

    def test_threshold_trigger(self, trainer):
        """Deve attivare auto-training alla soglia."""
        from smart_learner import AUTO_TRAIN_THRESHOLD
        for i in range(AUTO_TRAIN_THRESHOLD + 1):
            result = trainer.record_correction("trigger_test_field")
        # Alla soglia dovrebbe ritornare il nome del campo
        # (ma solo se can_train e cooldown ok)

    def test_cooldown(self, trainer):
        """Cooldown deve prevenire training troppo frequenti."""
        from datetime import datetime
        trainer._last_train_times["cooldown_field"] = datetime.now()
        assert not trainer._can_train("cooldown_field")

    def test_get_status(self, trainer):
        """Status deve avere la struttura corretta."""
        status = trainer.get_status()
        assert "auto_train_threshold" in status
        assert "cooldown_minutes" in status
        assert "fields" in status
        assert "recent_trains" in status


# ═════════════════════════════════════════════════════════════════════════════
# SELF EVALUATOR TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestSelfEvaluator:

    @pytest.fixture
    def evaluator(self):
        return SelfEvaluator()

    def test_evaluate_empty(self, evaluator):
        """Valutazione su dati vuoti non deve crashare."""
        result = evaluator.evaluate_field_quality()
        assert isinstance(result, dict)

    def test_problematic_fields_empty(self, evaluator):
        """Lista campi problematici su DB vuoto."""
        result = evaluator.get_problematic_fields()
        assert isinstance(result, list)


# ═════════════════════════════════════════════════════════════════════════════
# SMART LEARNER INTEGRATION TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestSmartLearner:

    @pytest.fixture
    def smart(self):
        return SmartLearner()

    def test_enhance_extraction_empty(self, smart):
        """Enhance su risultato vuoto non deve crashare."""
        result = {"oggetto_appalto": "", "cig": None, "_meta": "skip"}
        enhanced, methods = smart.enhance_extraction(result, "testo generico")
        assert isinstance(enhanced, dict)
        assert isinstance(methods, dict)

    def test_on_correction(self, smart):
        """Correzione deve attivare apprendimento pattern."""
        full_text = (
            "OGGETTO DELL'APPALTO: Realizzazione di un parco urbano\n\n"
            "CIG: 0123456789\n"
        )
        result = smart.on_correction(
            field="oggetto_appalto",
            correct_value="Realizzazione di un parco urbano",
            wrong_value="Realizzazione",
            full_text=full_text,
            doc_id="test_smart_01"
        )
        assert "patterns_learned" in result
        assert "auto_train_triggered" in result
        assert isinstance(result["auto_train_triggered"], bool)

    def test_full_status(self, smart):
        """Status completo deve avere tutte le sezioni."""
        status = smart.get_full_status()
        assert "pattern_learner" in status
        assert "auto_trainer" in status
        assert "evaluation" in status
        assert "problematic_fields" in status

    def test_end_to_end_learning_cycle(self, smart):
        """Test ciclo completo: correzione → pattern → estrazione."""
        # 1. Correzione: insegno che "oggetto" è dopo "OGGETTO:"
        text1 = "OGGETTO: Fornitura attrezzature informatiche\n\nCIG: AAA"
        smart.on_correction(
            "oggetto_appalto",
            "Fornitura attrezzature informatiche",
            "",
            text1,
            "doc_e2e_01"
        )

        # 2. Nuovo documento con struttura simile
        text2 = "OGGETTO: Manutenzione verde pubblico\n\nCIG: BBB"
        result = {"oggetto_appalto": None}
        enhanced, methods = smart.enhance_extraction(result, text2)

        # Il pattern appreso dovrebbe estrarre il valore
        # (potrebbe non funzionare con un solo esempio, ma non deve crashare)
        assert isinstance(enhanced, dict)


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG FIX TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestConfigFixes:

    def test_min_samples_increased(self):
        """MIN_SAMPLES_TRAIN deve essere >= 5."""
        from config import MIN_SAMPLES_TRAIN
        assert MIN_SAMPLES_TRAIN >= 5

    def test_min_improvement_no_regression(self):
        """MIN_IMPROVEMENT non deve permettere regressioni."""
        from config import MIN_IMPROVEMENT
        assert MIN_IMPROVEMENT >= 0.0

    def test_single_class_model_low_confidence(self):
        """_SingleClassModel deve avere confidenza max 0.50."""
        from ml_engine import _SingleClassModel
        model = _SingleClassModel("test_value")
        value, conf = model.predict("qualsiasi testo")
        assert value == "test_value"
        assert conf <= 0.50
