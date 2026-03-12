"""
AppaltoAI — Pipeline di Estrazione a 9 Fasi
═════════════════════════════════════════════
Orchestratore sottile: coordina le fasi della pipeline delegando
a moduli specializzati per ogni responsabilità.

  Fase 1:  Upload + Hash Dedup
  Fase 2:  Parsing PDF (pdf_parser)
  Fase 3:  Estrazione deterministica (rules_extractor) + NLP (nlp_classifier)
  Fase 4:  ML Enhancement (ml_engine) + Pattern Learning (smart_learner)
  Fase 4b: Adaptive Enhancement (adaptive_learner) — regole auto-generate + similarità
  Fase 5:  Costruzione JSON (json_builder)
  Fase 6:  Validazione (schemas) + Salvataggio
  Fase 6b: Post-Extraction Learning — il sistema impara da ogni documento
  Fase 7:  Gestione correzioni → dataset annotato proprietario
  Fase 8:  Retraining supervisionato (ml_engine)
  Fase 9:  Versionamento modelli con rollback
"""

import hashlib
import io
import json
import os
import pickle
import shutil
import tempfile
import time
from datetime import datetime
from typing import Optional, Dict, Tuple

from config import BASE_DIR, DATA_DIR, MODEL_DIR, MIN_SAMPLES_TRAIN
from database import get_connection, init_main_tables
from utils import find_value_context
from log_config import get_logger
from field_registry import registry
from pdf_parser import parse_pdf, get_text_with_tables, get_page_for_text, ParsedDocument
from schemas import full_validation
from ml_engine import ml_engine as ml
from smart_learner import smart_learner
from extractors import (
    extract_rules_based as disciplinari_extract,
    flatten_for_pipeline as disciplinari_flatten,
    extract_text_from_pdf as disciplinari_parse_pdf,
)
from json_builder import build_output, build_output_with_methods
from rules_extractor import RulesExtractor
from nlp_classifier import NLPClassifier
from adaptive_learner import adaptive_learner

logger = get_logger("pipeline")


# ═════════════════════════════════════════════════════════════════════════════
# FASE 1: UPLOAD + HASH DEDUP
# ═════════════════════════════════════════════════════════════════════════════

def compute_hash(content: bytes | str) -> str:
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def check_duplicate(text_hash: str) -> Optional[dict]:
    try:
        with get_connection(readonly=True) as conn:
            row = conn.execute(
                "SELECT id, filename, extracted_json, corrected_json "
                "FROM documents WHERE text_hash=? ORDER BY upload_date DESC LIMIT 1",
                (text_hash,)
            ).fetchone()
            if row and row[2]:
                result = json.loads(row[2])
                result["_cached"] = True
                result["_cached_doc_id"] = row[0]
                result["_cached_filename"] = row[1]
                if row[3]:
                    corrected = json.loads(row[3])
                    for k, v in corrected.items():
                        if v is not None:
                            result[k] = v
                            result.setdefault("_methods", {})[k] = "corrected"
                return result
    except Exception:
        logger.warning("Errore check_duplicate per hash %s", text_hash[:16], exc_info=True)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPALE — Orchestratore
# ═════════════════════════════════════════════════════════════════════════════

class Pipeline:
    """Orchestratore della pipeline a 9 fasi.
    Delega a moduli specializzati per ogni responsabilità."""

    def __init__(self):
        self.rules = RulesExtractor()
        self.nlp = NLPClassifier()
        self._last_parsed: Dict[str, ParsedDocument] = {}
        init_main_tables()

    # ═══════════════════════════════════════════════════════════════
    # FASI 1-6: Estrazione completa
    # ═══════════════════════════════════════════════════════════════

    def process_pdf(self, pdf_bytes: bytes, filename: str) -> dict:
        """Pipeline completa per PDF."""
        t_start = time.time()

        # FASE 1: Hash dedup
        pdf_hash = compute_hash(pdf_bytes)
        cached = check_duplicate(pdf_hash)
        if cached:
            cached["_pipeline_phase"] = "cached"
            cached["_pipeline_time_ms"] = round((time.time() - t_start) * 1000, 1)
            return cached

        # FASE 2: Parse PDF
        text, parsed = self._parse_pdf(pdf_bytes, filename)
        if not text or len(text.strip()) < 50:
            return {"error": "Impossibile estrarre testo dal PDF", "_pipeline_phase": "parse_failed"}

        # FASI 3-6
        result = self._extract_and_build(text, filename, pdf_bytes=pdf_bytes)

        # Aggiungi metadati PDF
        if parsed and parsed.pages:
            page_sources = {}
            for field, value in result.items():
                if field.startswith('_') or value is None or value == "" or isinstance(value, (bool, dict, list)):
                    continue
                page = get_page_for_text(parsed, str(value))
                if page:
                    page_sources[field] = page
            result["_page_sources"] = page_sources
            result["_total_pages"] = parsed.total_pages
            result["_is_native_pdf"] = parsed.is_native
            result["_parser_used"] = parsed.parser_used
            result["_pdf_metadata"] = parsed.metadata
            result["_pdf_warnings"] = parsed.warnings
            result["_tables_found"] = len(parsed.tables_json)
            result["_tables_json"] = parsed.tables_json[:20]
            result["_chunks_count"] = len(parsed.chunks)

        # Salva con hash
        self._save_document(result["_doc_id"], filename, text, result, pdf_hash)
        result["_pipeline_phase"] = "complete"
        result["_pipeline_time_ms"] = round((time.time() - t_start) * 1000, 1)
        return result

    def process_text(self, text: str, filename: str = "input.txt") -> dict:
        """Pipeline per testo grezzo."""
        t_start = time.time()

        text_hash = compute_hash(text)
        cached = check_duplicate(text_hash)
        if cached:
            cached["_pipeline_phase"] = "cached"
            return cached

        result = self._extract_and_build(text, filename)
        self._save_document(result["_doc_id"], filename, text, result, text_hash)
        result["_pipeline_phase"] = "complete"
        result["_pipeline_time_ms"] = round((time.time() - t_start) * 1000, 1)
        return result

    def _extract_and_build(self, text: str, filename: str, pdf_bytes: bytes = None) -> dict:
        """Fasi 2-6: Disciplinari/Rules → NLP → ML → Build JSON → Validate."""
        # FASE 2: Estrazione deterministica avanzata (extract_disciplinari)
        try:
            if pdf_bytes:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(pdf_bytes)
                    tmp_path = tmp.name
                try:
                    disc_text = disciplinari_parse_pdf(tmp_path)
                finally:
                    os.unlink(tmp_path)
                nested_result = disciplinari_extract(disc_text)
            else:
                nested_result = disciplinari_extract(text)

            logger.debug("Nested extraction: %d sections", len(nested_result))
            rules_result, methods = build_output_with_methods(nested_result)
            logger.debug("Build output: %d keys, %d methods", len(rules_result), len(methods))
            snippets = {}
        except Exception:
            logger.warning("Disciplinari extractor failed, falling back to rules", exc_info=True)
            rules_result, snippets, methods = self.rules.extract(text)

        # FASE 3: NLP classificazione
        nlp_tipo = self.nlp.classify_procedure(text)
        if nlp_tipo and nlp_tipo != "Non specificata":
            if not rules_result.get("tipologia_appalto"):
                rules_result["tipologia_appalto"] = nlp_tipo
        nlp_criterio = self.nlp.classify_criterio(text)
        if nlp_criterio and nlp_criterio != "Non specificato":
            tipologia = rules_result.get("tipologia_appalto", "")
            if tipologia and nlp_criterio.lower() not in tipologia.lower():
                rules_result["tipologia_appalto"] = f"{tipologia} - {nlp_criterio}"
            elif not tipologia:
                rules_result["tipologia_appalto"] = nlp_criterio
        if "tipologia_appalto" not in methods and rules_result.get("tipologia_appalto"):
            methods["tipologia_appalto"] = "rules"

        # FASE 3b: ML Engine enhancement
        rules_result, ml_methods = ml.enhance_result(rules_result, text)
        methods.update(ml_methods)
        for key in ml_methods:
            if key not in snippets:
                ctx = find_value_context(text, str(rules_result.get(key, "")))
                if ctx:
                    snippets[key] = ctx

        # FASE 3c: Pattern appresi (smart_learner)
        try:
            rules_result, pattern_methods = smart_learner.enhance_extraction(
                rules_result, text
            )
            methods.update(pattern_methods)
            for key in pattern_methods:
                if key not in snippets:
                    ctx = find_value_context(text, str(rules_result.get(key, "")))
                    if ctx:
                        snippets[key] = ctx
        except Exception:
            pass

        # FASE 4b: Adaptive Enhancement — regole auto-generate + similarità doc
        try:
            rules_result, adaptive_methods = adaptive_learner.enhance_result(
                rules_result, text, methods
            )
            methods.update(adaptive_methods)
            for key in adaptive_methods:
                if key not in snippets:
                    ctx = find_value_context(text, str(rules_result.get(key, "")))
                    if ctx:
                        snippets[key] = ctx
        except Exception:
            logger.debug("Adaptive enhancement failed", exc_info=True)

        # FASE 5: Costruzione JSON
        doc_id = hashlib.sha256((filename + text[:500]).encode()).hexdigest()[:16]
        result = dict(rules_result)
        result["_snippets"] = snippets
        result["_methods"] = methods
        result["_doc_id"] = doc_id
        result["_model_version"] = self._get_active_model_version()

        # Confidence
        empty_objects = [
            {"quota_fissa_65_perc": False, "quota_ribassabile_35_perc": False},
            {"obbligatorio": False, "note": None},
            {"partecipazione_tutti_lotti": False, "offerta_identica": False,
             "max_lotti_aggiudicazione": None, "medesima_forma_giuridica": False}
        ]
        total = sum(1 for k in result if not k.startswith('_'))
        filled = sum(1 for k, v in result.items()
                     if not k.startswith('_') and v not in [None, [], {}, False, "", 0]
                     and v not in empty_objects)
        result["_confidence"] = round(filled / max(total, 1) * 100, 1)
        result["_extraction_method"] = "hybrid" if ml_methods else "rules"
        result["_timestamp"] = datetime.now().isoformat()
        result["_text_length"] = len(text)
        result["_filename"] = filename

        # FASE 5b: Validazione
        try:
            validation = full_validation(result)
            result["_coherence"] = validation["coherence"]
            result["_validation_warnings"] = validation["warnings"]
        except Exception:
            pass

        # FASE 6b: Post-Extraction Learning — il sistema impara da ogni doc
        try:
            adaptive_learner.post_extraction_learn(
                doc_id, filename, text, result, methods
            )
            result["_adaptive_learning"] = True
        except Exception:
            logger.debug("Adaptive post-learning failed", exc_info=True)
            result["_adaptive_learning"] = False

        # FASE 6c: Raccolta Dati ML
        try:
            n_collected = ml.collect_from_extraction(text, result, methods, doc_id=doc_id)
            result["_ml_data_collected"] = n_collected
        except Exception:
            pass

        # ML status
        try:
            ml_status = ml.get_status()
            result["_ml_models_active"] = ml_status["active_models"]
            result["_ml_training_data"] = ml_status["total_training_data"]
        except Exception:
            pass

        return result

    def _parse_pdf(self, pdf_bytes: bytes, filename: str) -> Tuple[str, Optional[ParsedDocument]]:
        parsed = None
        try:
            parsed = parse_pdf(pdf_bytes, filename)
            self._last_parsed[filename] = parsed
            text = get_text_with_tables(parsed)
            if text and len(text.strip()) > 50:
                return text, parsed
        except Exception:
            pass
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
            if text.strip():
                return text, parsed
        except Exception:
            pass
        return "", None

    def get_last_parsed(self, filename: str) -> Optional[ParsedDocument]:
        return self._last_parsed.get(filename)

    # ═══════════════════════════════════════════════════════════════
    # FASE 7: CORREZIONI → DATASET ANNOTATO PROPRIETARIO
    # ═══════════════════════════════════════════════════════════════

    def record_correction(self, doc_id: str, field: str, original: str,
                          corrected: str, snippet: str = "") -> dict:
        """Registra correzione → annotated training example."""
        with get_connection() as conn:
            c = conn.cursor()
            row_text = c.execute("SELECT full_text FROM documents WHERE id=?", (doc_id,)).fetchone()
            full_text = row_text[0] if row_text and row_text[0] else ""
            training_snippet = snippet
            if full_text:
                ctx = find_value_context(full_text, corrected, window=500)
                if ctx and len(ctx) > 20:
                    training_snippet = ctx
                elif original:
                    ctx = find_value_context(full_text, original, window=500)
                    if ctx and len(ctx) > 20:
                        training_snippet = ctx
            if not training_snippet:
                training_snippet = full_text[:1500]
            c.execute(
                "INSERT INTO feedback_log (doc_id, field, original, corrected) VALUES (?,?,?,?)",
                (doc_id, field, original, corrected),
            )
            c.execute(
                "INSERT INTO training_samples (field, text_snippet, correct_value, wrong_value, source) "
                "VALUES (?,?,?,?,?)",
                (field, training_snippet[:2000], corrected, original, "correction"),
            )
            row = c.execute("SELECT corrected_json FROM documents WHERE id=?", (doc_id,)).fetchone()
            cd = json.loads(row[0]) if row and row[0] else {}
            cd[field] = corrected
            c.execute(
                "UPDATE documents SET corrected_json=? WHERE id=?",
                (json.dumps(cd, ensure_ascii=False), doc_id),
            )

        # Feed to ML Engine
        try:
            ml.add_correction(
                field, training_snippet, corrected,
                wrong_value=original, doc_id=doc_id
            )
        except Exception:
            pass

        # Feed to Smart Learner
        smart_learning_result = None
        try:
            smart_learning_result = smart_learner.on_correction(
                field, corrected, original, full_text, doc_id
            )
        except Exception:
            pass

        # Feed to Adaptive Learner
        adaptive_result = None
        try:
            adaptive_result = adaptive_learner.on_correction(
                doc_id, field, corrected, original, full_text
            )
        except Exception:
            pass

        count = self._get_sample_count(field)
        ml_data = ml.data.get_all_fields()
        ml_count = ml_data.get(field, 0)
        can_train = ml_count >= MIN_SAMPLES_TRAIN

        patterns_learned = 0
        auto_trained = False
        if smart_learning_result:
            patterns_learned = len(smart_learning_result.get("patterns_learned", []))
            auto_trained = smart_learning_result.get("auto_train_triggered", False)

        auto_rules_gen = 0
        if adaptive_result:
            auto_rules_gen = adaptive_result.get("new_rules_generated", 0)

        msg = f"Correzione salvata ({ml_count} campioni ML)."
        if patterns_learned > 0:
            msg += f" {patterns_learned} pattern strutturali appresi."
        if auto_rules_gen > 0:
            msg += f" {auto_rules_gen} regole auto-generate."
        if auto_trained:
            train_status = smart_learning_result.get("auto_train_result", {}).get("status", "")
            msg += f" Auto-training eseguito ({train_status})."
        elif can_train:
            msg += " Puoi addestrare il modello!"
        else:
            msg += f" Servono {MIN_SAMPLES_TRAIN - ml_count} campioni in più."

        return {
            "status": "ok", "field": field,
            "sample_count": count,
            "ml_samples": ml_count,
            "can_train": can_train,
            "patterns_learned": patterns_learned,
            "auto_rules_generated": auto_rules_gen,
            "auto_trained": auto_trained,
            "message": msg,
        }

    # ═══════════════════════════════════════════════════════════════
    # FASE 8: RETRAINING SUPERVISIONATO
    # ═══════════════════════════════════════════════════════════════

    def train_field(self, field: str) -> dict:
        return ml.train_field(field)

    def train_all(self) -> dict:
        return ml.train_all()

    # ═══════════════════════════════════════════════════════════════
    # FASE 9: MODEL VERSIONING + ROLLBACK
    # ═══════════════════════════════════════════════════════════════

    def get_ml_status(self) -> dict:
        return ml.get_status()

    def get_ml_quality(self) -> dict:
        return ml.get_quality_report()

    def rollback_model(self, field: str) -> dict:
        backup = MODEL_DIR / f"model_{field}_prev.pkl"
        path = MODEL_DIR / f"model_{field}.pkl"
        if not backup.exists():
            return {"status": "error", "message": f"Nessun backup per '{field}'"}
        shutil.copy2(backup, path)
        with open(path, "rb") as f:
            self.nlp.models[field] = pickle.load(f)
        with get_connection() as conn:
            conn.execute("UPDATE model_versions SET is_active=0 WHERE field=?", (field,))
            conn.execute(
                "UPDATE model_versions SET is_active=1 "
                "WHERE field=? AND id=(SELECT MAX(id)-1 FROM model_versions WHERE field=?)",
                (field, field),
            )
        return {"status": "ok", "message": f"Rollback '{field}' completato"}

    def get_model_versions(self, field: str = None) -> list:
        with get_connection(readonly=True) as conn:
            if field:
                rows = conn.execute(
                    "SELECT field, version, accuracy, samples_count, trained_at, is_active, notes "
                    "FROM model_versions WHERE field=? ORDER BY version DESC", (field,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT field, version, accuracy, samples_count, trained_at, is_active, notes "
                    "FROM model_versions ORDER BY trained_at DESC"
                ).fetchall()
        return [{"field": r[0], "version": r[1], "accuracy": r[2], "samples": r[3],
                 "trained_at": r[4], "is_active": bool(r[5]), "notes": r[6]} for r in rows]

    def _get_active_model_version(self) -> str:
        with get_connection(readonly=True) as conn:
            rows = conn.execute(
                "SELECT field, version FROM model_versions WHERE is_active=1"
            ).fetchall()
        if not rows:
            return "rules-only"
        return ", ".join(f"{r[0]}:v{r[1]}" for r in rows)

    # ═══════════════════════════════════════════════════════════════
    # CORRECTIONS CRUD
    # ═══════════════════════════════════════════════════════════════

    def get_corrections(self, limit: int = 200) -> list:
        with get_connection(readonly=True) as conn:
            c = conn.cursor()
            rows = c.execute("""
                SELECT f.id, f.doc_id, f.field, f.original, f.corrected, f.timestamp,
                       t.id as sample_id, t.text_snippet
                FROM feedback_log f
                LEFT JOIN training_samples t
                    ON t.field = f.field AND t.correct_value = f.corrected AND t.wrong_value = f.original
                ORDER BY f.timestamp DESC LIMIT ?
            """, (limit,)).fetchall()
            corrections = []
            seen = set()
            for r in rows:
                key = (r[0], r[6])
                if key in seen:
                    continue
                seen.add(key)
                corrections.append({
                    "id": r[0], "doc_id": r[1], "field": r[2], "original": r[3],
                    "corrected": r[4], "timestamp": r[5], "sample_id": r[6],
                    "snippet": (r[7] or "")[:300],
                })
            manual = c.execute("""
                SELECT t.id, t.field, t.text_snippet, t.correct_value, t.wrong_value, t.created_at, t.source
                FROM training_samples t
                WHERE NOT EXISTS (
                    SELECT 1 FROM feedback_log f
                    WHERE f.field=t.field AND f.corrected=t.correct_value
                )
                ORDER BY t.created_at DESC LIMIT ?
            """, (limit,)).fetchall()
            for r in manual:
                corrections.append({
                    "id": None, "doc_id": None, "field": r[1], "original": r[4] or "",
                    "corrected": r[3], "timestamp": r[5], "sample_id": r[0],
                    "snippet": (r[2] or "")[:300], "source": r[6] or "training_sample",
                })
        corrections.sort(key=lambda x: x.get("timestamp") or "", reverse=True)
        return corrections[:limit]

    def update_correction(self, correction_id=None, sample_id=None, data=None) -> dict:
        if not data:
            return {"status": "error", "message": "Nessun dato"}
        nc = data.get("corrected")
        nf = data.get("field")
        ns = data.get("snippet")
        updated = 0
        with get_connection() as conn:
            c = conn.cursor()
            if correction_id:
                if nc:
                    c.execute("UPDATE feedback_log SET corrected=? WHERE id=?", (nc, correction_id))
                    updated += 1
                if nf:
                    c.execute("UPDATE feedback_log SET field=? WHERE id=?", (nf, correction_id))
            if sample_id:
                if nc:
                    c.execute("UPDATE training_samples SET correct_value=? WHERE id=?", (nc, sample_id))
                    updated += 1
                if ns:
                    c.execute("UPDATE training_samples SET text_snippet=? WHERE id=?", (ns[:2000], sample_id))
                if nf:
                    c.execute("UPDATE training_samples SET field=? WHERE id=?", (nf, sample_id))
        return {"status": "ok", "updated": updated}

    def delete_correction(self, correction_id=None, sample_id=None) -> dict:
        deleted = 0
        with get_connection() as conn:
            c = conn.cursor()
            if correction_id:
                row = c.execute(
                    "SELECT field, corrected, original FROM feedback_log WHERE id=?",
                    (correction_id,),
                ).fetchone()
                if row:
                    c.execute("DELETE FROM feedback_log WHERE id=?", (correction_id,))
                    c.execute(
                        "DELETE FROM training_samples "
                        "WHERE field=? AND correct_value=? AND (wrong_value=? OR wrong_value IS NULL)",
                        row,
                    )
                    deleted += 1
            if sample_id:
                c.execute("DELETE FROM training_samples WHERE id=?", (sample_id,))
                deleted += 1
        return {"status": "ok", "deleted": deleted}

    # ═══════════════════════════════════════════════════════════════
    # STATS & HISTORY
    # ═══════════════════════════════════════════════════════════════

    def get_stats(self) -> dict:
        with get_connection(readonly=True) as conn:
            c = conn.cursor()
            docs = c.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            corrections = c.execute("SELECT COUNT(*) FROM feedback_log").fetchone()[0]
            samples = c.execute(
                "SELECT field, COUNT(*) FROM training_samples GROUP BY field"
            ).fetchall()
            models = c.execute(
                "SELECT field, version, accuracy, samples_count, trained_at, is_active "
                "FROM model_versions ORDER BY trained_at DESC"
            ).fetchall()
        return {
            "total_documents": docs,
            "total_corrections": corrections,
            "training_samples": {s[0]: s[1] for s in samples},
            "trained_models": [
                {"field": m[0], "version": m[1], "accuracy": m[2], "samples": m[3],
                 "trained_at": m[4], "is_active": bool(m[5])}
                for m in models
            ],
            "loaded_ml_models": list(self.nlp.models.keys()),
            "ml_engine": ml.get_status(),
        }

    def get_history(self) -> list:
        with get_connection(readonly=True) as conn:
            rows = conn.execute(
                "SELECT id, filename, upload_date, extracted_json, model_version "
                "FROM documents ORDER BY upload_date DESC LIMIT 50"
            ).fetchall()
        result = []
        for r in rows:
            try:
                data = json.loads(r[3]) if r[3] else {}
                result.append({
                    "id": r[0], "filename": r[1], "date": r[2],
                    "oggetto": data.get("oggetto_appalto", "N/D"),
                    "importo": data.get("importo_totale") or data.get("importo_base_gara", "N/D"),
                    "stazione": data.get("stazione_appaltante", "N/D"),
                    "confidence": data.get("_confidence", 0),
                    "lotti": data.get("numero_lotti"),
                    "procedura": data.get("tipo_procedura", "N/D"),
                    "model_version": r[4] or "rules-only",
                })
            except Exception:
                pass
        return result

    def get_document_text(self, doc_id: str) -> Optional[dict]:
        with get_connection(readonly=True) as conn:
            row = conn.execute(
                "SELECT full_text, extracted_json, corrected_json FROM documents WHERE id=?",
                (doc_id,),
            ).fetchone()
        if row:
            return {
                "text": row[0] or "",
                "extracted": json.loads(row[1]) if row[1] else {},
                "corrected": json.loads(row[2]) if row[2] else {},
            }
        return None

    def get_corrections_stats(self) -> dict:
        with get_connection(readonly=True) as conn:
            c = conn.cursor()
            stats = c.execute(
                "SELECT field, COUNT(*) FROM feedback_log GROUP BY field ORDER BY COUNT(*) DESC"
            ).fetchall()
            total = c.execute("SELECT COUNT(*) FROM feedback_log").fetchone()[0]
            samples = c.execute("SELECT COUNT(*) FROM training_samples").fetchone()[0]
        return {
            "total_corrections": total,
            "total_samples": samples,
            "by_field": {s[0]: s[1] for s in stats},
        }

    def _save_document(self, doc_id, filename, text, extracted, text_hash=None):
        if not text_hash:
            text_hash = hashlib.sha256(text.encode()).hexdigest()
        with get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO documents "
                "(id, filename, text_hash, full_text, upload_date, extracted_json, model_version) "
                "VALUES (?,?,?,?,?,?,?)",
                (doc_id, filename, text_hash, text, datetime.now().isoformat(),
                 json.dumps(extracted, ensure_ascii=False, default=str),
                 extracted.get("_model_version", "rules-only")),
            )

    def _get_sample_count(self, field: str) -> int:
        with get_connection(readonly=True) as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM training_samples WHERE field=?", (field,)
            ).fetchone()[0]


# ── Singleton globale ──────────────────────────────────────────────────
pipeline = Pipeline()
