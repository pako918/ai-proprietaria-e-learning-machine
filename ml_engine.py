"""
AppaltoAI — Machine Learning Engine
=====================================
Il modello impara schemi dai dati.
La qualità dei dati determina la qualità del modello.

Ciclo di apprendimento:
  Documenti → Estrazione → Correzioni Umane → Dati Annotati
    → Feature Engineering → Training Modelli → Predizioni Migliori

Componenti:
  - DataStore: raccolta e gestione dati di training con qualità
  - FieldModel: modello ML per campo con dual TF-IDF + LogisticRegression
  - MLEngine: orchestratore training/inference/monitoring
  - QualityMonitor: metriche qualità dati e modelli

Principi:
  - Il modello impara SCHEMI dai dati, non regole hardcoded
  - Più dati di qualità = modello migliore
  - Feedback loop: estrazione → correzione → training → migliore estrazione
  - Trasparenza: metriche visibili per ogni modello
  - Supervised: l'utente controlla training e rollback
"""

import json
import pickle
import re
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline, FeatureUnion
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from config import (
    MODEL_DIR, DB_PATH,
    MIN_SAMPLES_TRAIN, MIN_SAMPLES_CV, MIN_IMPROVEMENT,
    DEFAULT_CONFIDENCE_THRESHOLD, ML_OVERRIDE_THRESHOLD,
    DATA_QUALITY_WEIGHTS,
)
from database import get_connection, init_ml_tables
from utils import find_value_context
from log_config import get_logger

logger = get_logger("ml_engine")


# ═════════════════════════════════════════════════════════════════════
# DATA STORE — Il cuore del sistema: gestione dati di training
# ═════════════════════════════════════════════════════════════════════

class DataStore:
    """Gestisce la raccolta, qualità e recupero dei dati di training.

    I dati sono il carburante del modello. Questo componente:
    - Raccoglie esempi da ogni documento processato
    - Assegna punteggi di qualità a ogni esempio
    - Deduplica automaticamente
    - Traccia la provenienza (correzione umana, regole, auto)
    - Fornisce metriche sulla qualità del dataset
    """

    def __init__(self):
        init_ml_tables()

    # ── Aggiunta dati ──────────────────────────────────────────────

    def add_example(self, field: str, text: str, value: str,
                    source: str = "auto", doc_id: str = "",
                    wrong_value: str = None) -> int:
        """Aggiunge un esempio di training con punteggio di qualità.

        La qualità dipende dalla fonte:
        - correction (1.0): correzione umana
        - manual (0.95): aggiunto manualmente
        - rules_validated (0.60): regex + validazione formato
        - rules_raw (0.35): solo regex
        - auto (0.20): auto-raccolto
        """
        if not text or not value or len(text.strip()) < 10:
            return -1

        quality = DATA_QUALITY_WEIGHTS.get(source, 0.3)
        # Bonus qualità per snippet più lunghi e informativi
        if len(text) > 200:
            quality = min(1.0, quality + 0.05)
        if len(text) > 500:
            quality = min(1.0, quality + 0.05)

        with get_connection() as conn:
            c = conn.cursor()

            # Anti-duplicato: (campo, valore, primi 200 char dello snippet)
            existing = c.execute(
                "SELECT id FROM ml_training_data "
                "WHERE field=? AND correct_value=? AND substr(text_snippet,1,200)=? AND is_active=1",
                (field, value, text[:200])
            ).fetchone()
            if existing:
                return existing[0]

            c.execute(
                "INSERT INTO ml_training_data "
                "(field, text_snippet, correct_value, wrong_value, source, quality_score, doc_id) "
                "VALUES (?,?,?,?,?,?,?)",
                (field, text[:3000], value, wrong_value, source, quality, doc_id)
            )
            return c.lastrowid

    def add_correction(self, field: str, text: str, correct_value: str,
                       wrong_value: str = None, doc_id: str = "") -> int:
        """Aggiunge una correzione umana — la fonte dati più preziosa."""
        return self.add_example(field, text, correct_value,
                               source="correction", doc_id=doc_id,
                               wrong_value=wrong_value)

    def add_document_examples(self, doc_text: str, extracted: dict,
                              methods: dict, doc_id: str = "",
                              validated_fields: set = None):
        """Auto-raccolta dati da un documento processato.
        Ogni campo estratto diventa un potenziale esempio di training."""
        count = 0
        for field, value in extracted.items():
            if field.startswith('_') or value is None or value == "":
                continue
            if isinstance(value, (bool, dict, list)):
                continue
            if value == 0 or value is False:
                continue

            method = methods.get(field, "")
            context = self._find_context(doc_text, str(value))
            if not context or len(context) < 30:
                continue

            # Qualità basata sul metodo
            is_validated = validated_fields and field in validated_fields
            if method == "corrected":
                source = "correction"
            elif "ml" in str(method) and is_validated:
                source = "rules_validated"  # ML + validated
            elif method == "rules" and is_validated:
                source = "rules_validated"
            elif method == "rules":
                source = "rules_raw"
            else:
                source = "auto"

            self.add_example(field, context, str(value),
                           source=source, doc_id=doc_id)
            count += 1
        return count

    # ── Recupero dati ──────────────────────────────────────────────

    def get_training_data(self, field: str,
                         min_quality: float = 0.0) -> List[Tuple[str, str, float]]:
        """Recupera dati di training per un campo: [(testo, valore, qualità), ...]"""
        with get_connection(readonly=True) as conn:
            rows = conn.execute(
                "SELECT text_snippet, correct_value, quality_score FROM ml_training_data "
                "WHERE field=? AND is_active=1 AND quality_score >= ? "
                "ORDER BY quality_score DESC",
                (field, min_quality)
            ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def get_all_fields(self) -> Dict[str, int]:
        """Tutti i campi con conteggio campioni."""
        with get_connection(readonly=True) as conn:
            rows = conn.execute(
                "SELECT field, COUNT(*) FROM ml_training_data "
                "WHERE is_active=1 GROUP BY field ORDER BY COUNT(*) DESC"
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def get_data_quality(self, field: str = None) -> dict:
        """Metriche di qualità del dataset."""
        with get_connection(readonly=True) as conn:
            c = conn.cursor()

            if field:
                where = "WHERE field=? AND is_active=1"
                params = (field,)
            else:
                where = "WHERE is_active=1"
                params = ()

            total = c.execute(
                f"SELECT COUNT(*) FROM ml_training_data {where}", params
            ).fetchone()[0]
            avg_quality = c.execute(
                f"SELECT AVG(quality_score) FROM ml_training_data {where}", params
            ).fetchone()[0] or 0

            sources = c.execute(
                f"SELECT source, COUNT(*), AVG(quality_score) "
                f"FROM ml_training_data {where} GROUP BY source", params
            ).fetchall()

            fields_data = c.execute(
                f"SELECT field, COUNT(*), AVG(quality_score) "
                f"FROM ml_training_data {where} GROUP BY field ORDER BY COUNT(*) DESC", params
            ).fetchall()

            diversity = c.execute(
                f"SELECT field, COUNT(DISTINCT correct_value) "
                f"FROM ml_training_data {where} GROUP BY field", params
            ).fetchall()

        return {
            "total_examples": total,
            "avg_quality": round(avg_quality, 3),
            "by_source": {
                s[0]: {"count": s[1], "avg_quality": round(s[2], 3)}
                for s in sources
            },
            "by_field": {
                f[0]: {"count": f[1], "avg_quality": round(f[2], 3)}
                for f in fields_data
            },
            "value_diversity": {d[0]: d[1] for d in diversity},
        }

    def remove_example(self, example_id: int):
        """Disattiva un esempio (soft delete)."""
        with get_connection() as conn:
            conn.execute(
                "UPDATE ml_training_data SET is_active=0 WHERE id=?", (example_id,)
            )

    # ── Migrazione da vecchio schema ───────────────────────────────

    def migrate_from_old_tables(self) -> int:
        """Importa dati dalla vecchia tabella training_samples."""
        try:
            with get_connection() as conn:
                c = conn.cursor()
                rows = c.execute(
                    "SELECT field, text_snippet, correct_value, wrong_value, source, created_at "
                    "FROM training_samples"
                ).fetchall()
                migrated = 0
                for r in rows:
                    existing = c.execute(
                        "SELECT id FROM ml_training_data "
                        "WHERE field=? AND correct_value=? AND substr(text_snippet,1,200)=?",
                        (r[0], r[2], (r[1] or "")[:200])
                    ).fetchone()
                    if not existing:
                        source = r[4] or "manual"
                        quality = DATA_QUALITY_WEIGHTS.get(source, 0.5)
                        c.execute(
                            "INSERT INTO ml_training_data "
                            "(field, text_snippet, correct_value, wrong_value, source, "
                            "quality_score, created_at) VALUES (?,?,?,?,?,?,?)",
                            (r[0], r[1], r[2], r[3], source, quality, r[5])
                        )
                        migrated += 1
                return migrated
        except Exception:
            return 0

    # ── Utilità ────────────────────────────────────────────────────

    def _find_context(self, text: str, value: str, window: int = 500) -> str:
        """Trova il contesto testuale intorno a un valore nel documento."""
        return find_value_context(text, value, window=window)


# ═════════════════════════════════════════════════════════════════════
# FIELD MODEL — Modello ML per singolo campo
# ═════════════════════════════════════════════════════════════════════

class _SingleClassModel:
    """Modello triviale per campi con un solo valore noto."""
    def __init__(self, value):
        self.value = value

    def predict(self, text):
        return self.value, 0.90


class FieldModel:
    """Modello ML per l'estrazione di un campo specifico.

    Impara schemi dal testo per predire il valore corretto.
    Usa dual TF-IDF (word + char n-grams) per features robuste
    e LogisticRegression calibrata per confidenza affidabile.

    Il modello:
    - Impara quali pattern testuali corrispondono a quali valori
    - Più dati = features migliori = predizioni più accurate
    - Cross-validation per stimare la vera accuratezza
    - Confidence calibrata per sapere "quanto è sicuro"
    """

    def __init__(self, field_name: str, field_type: str = "text"):
        self.field = field_name
        self.field_type = field_type
        self.model = None       # sklearn Pipeline
        self.classes = []
        self.n_samples = 0
        self.metrics = {}
        self.trained_at = None
        self.version = 0

    def train(self, texts: List[str], values: List[str],
              qualities: List[float] = None) -> dict:
        """Addestra il modello sui dati. Ritorna metriche.

        Feature Engineering:
        - Word TF-IDF (1-2 gram): cattura significato semantico
          ("stazione appaltante", "importo base")
        - Char TF-IDF (2-5 gram): cattura pattern morfologici
          e varianti parziali (robusto per italiano)

        Modello: LogisticRegression con class_weight='balanced'
        - Gestisce classi sbilanciate
        - predict_proba per confidenza calibrata
        - Regolarizzazione L2 per evitare overfitting
        """
        t_start = time.time()
        n = len(texts)

        if n < MIN_SAMPLES_TRAIN:
            return {
                "status": "error",
                "message": f"Dati insufficienti: {n}/{MIN_SAMPLES_TRAIN} richiesti"
            }

        unique_values = list(set(values))
        self.classes = unique_values
        self.n_samples = n

        # ── Caso speciale: un solo valore unico ─────────────────
        if len(unique_values) == 1:
            self.model = _SingleClassModel(unique_values[0])
            self.metrics = {
                "accuracy": 1.0, "n_samples": n, "n_classes": 1,
                "classes": unique_values,
                "training_time_ms": round((time.time() - t_start) * 1000, 1),
            }
            self.trained_at = datetime.now().isoformat()
            self.version += 1
            return {"status": "ok", "metrics": self.metrics}

        # ── Pesi campioni dalla qualità dei dati ─────────────────
        sample_weight = None
        if qualities:
            sample_weight = np.array(qualities, dtype=np.float64) + 0.1

        # ── Feature Engineering: Dual TF-IDF ─────────────────────
        feature_union = FeatureUnion([
            ("word_tfidf", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                max_features=10000,
                sublinear_tf=True,
                min_df=1,
                max_df=1.0,
            )),
            ("char_tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                max_features=15000,
                sublinear_tf=True,
                min_df=1,
                max_df=1.0,
            )),
        ])

        # ── Cross-Validation per metriche reali ──────────────────
        cv_metrics = {}
        if n >= MIN_SAMPLES_CV and len(unique_values) >= 2:
            try:
                X_feat = feature_union.fit_transform(texts)
                min_class_count = min(Counter(values).values())
                n_splits = min(5, min_class_count, n // 2)
                if n_splits >= 2:
                    cv_model = LogisticRegression(
                        C=1.0, max_iter=2000, solver='lbfgs',
                        class_weight='balanced',
                    )
                    skf = StratifiedKFold(
                        n_splits=n_splits, shuffle=True, random_state=42
                    )
                    scores = cross_val_score(
                        cv_model, X_feat, values, cv=skf, scoring='accuracy'
                    )
                    cv_metrics = {
                        "cv_mean": float(np.mean(scores)),
                        "cv_std": float(np.std(scores)),
                        "cv_scores": [float(s) for s in scores],
                    }
            except Exception as e:
                logger.warning(f"CV failed for {self.field}: {e}")

        # ── Training finale su TUTTI i dati ──────────────────────
        classifier = LogisticRegression(
            C=1.0, max_iter=2000, solver='lbfgs',
            class_weight='balanced',
        )

        self.model = SkPipeline([
            ("features", feature_union),
            ("clf", classifier),
        ])

        if sample_weight is not None:
            self.model.fit(texts, values, clf__sample_weight=sample_weight)
        else:
            self.model.fit(texts, values)

        # ── Metriche sul training set ─────────────────────────────
        train_preds = self.model.predict(texts)
        train_acc = accuracy_score(values, train_preds)

        self.metrics = {
            "accuracy": float(train_acc),
            "n_samples": n,
            "n_classes": len(unique_values),
            "classes": unique_values[:20],  # Max 20 per serializzazione
            "training_time_ms": round((time.time() - t_start) * 1000, 1),
            **cv_metrics,
        }

        # F1 score
        try:
            avg = 'binary' if len(unique_values) == 2 else 'weighted'
            pos = unique_values[0] if len(unique_values) == 2 else None
            self.metrics["f1"] = float(
                f1_score(values, train_preds, average=avg, pos_label=pos)
            )
        except Exception:
            pass

        # Feature importance (top terms)
        try:
            self._extract_feature_importance()
        except Exception:
            pass

        self.trained_at = datetime.now().isoformat()
        self.version += 1
        return {"status": "ok", "metrics": self.metrics}

    def _extract_feature_importance(self):
        """Estrae le feature più importanti dal modello."""
        if not self.model or isinstance(self.model, _SingleClassModel):
            return
        try:
            clf = self.model.named_steps["clf"]
            features = self.model.named_steps["features"]
            feature_names = features.get_feature_names_out()
            if hasattr(clf, "coef_"):
                importances = np.abs(clf.coef_).mean(axis=0) if clf.coef_.ndim > 1 else np.abs(clf.coef_[0])
                top_idx = np.argsort(importances)[-20:][::-1]
                self.metrics["top_features"] = [
                    {"feature": feature_names[i], "importance": float(importances[i])}
                    for i in top_idx
                ]
        except Exception:
            pass

    def predict(self, text: str) -> Tuple[Optional[str], float]:
        """Predice il valore con confidenza. Ritorna (valore, confidenza)."""
        if self.model is None:
            return None, 0.0

        try:
            # Modello a classe singola
            if isinstance(self.model, _SingleClassModel):
                return self.model.predict(text)

            value = self.model.predict([text])[0]

            # Confidenza da predict_proba (calibrata)
            confidence = 0.5
            try:
                probas = self.model.predict_proba([text])[0]
                confidence = float(np.max(probas))
            except Exception:
                # Fallback: decision function
                try:
                    df = self.model.decision_function([text])
                    if hasattr(df, '__len__') and len(df.shape) > 1:
                        confidence = float(np.max(np.abs(df)))
                    else:
                        confidence = min(1.0, float(abs(df[0])) / 2.0)
                except Exception:
                    pass

            return str(value), confidence

        except Exception as e:
            logger.warning(f"Prediction failed for {self.field}: {e}")
            return None, 0.0

    def predict_top_k(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        """Predice i top-k valori con confidenza."""
        if self.model is None or isinstance(self.model, _SingleClassModel):
            v, c = self.predict(text)
            return [(v, c)] if v else []

        try:
            probas = self.model.predict_proba([text])[0]
            classes = self.model.classes_
            top_idx = np.argsort(probas)[-k:][::-1]
            return [(str(classes[i]), float(probas[i])) for i in top_idx if probas[i] > 0.05]
        except Exception:
            v, c = self.predict(text)
            return [(v, c)] if v else []

    def save(self, path: Path):
        """Salva modello su disco."""
        with open(path, "wb") as f:
            pickle.dump({
                "field": self.field,
                "field_type": self.field_type,
                "model": self.model,
                "classes": self.classes,
                "n_samples": self.n_samples,
                "metrics": self.metrics,
                "trained_at": self.trained_at,
                "version": self.version,
            }, f)

    @classmethod
    def load(cls, path: Path) -> 'FieldModel':
        """Carica modello da disco."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        fm = cls(data["field"], data.get("field_type", "text"))
        fm.model = data["model"]
        fm.classes = data.get("classes", [])
        fm.n_samples = data.get("n_samples", 0)
        fm.metrics = data.get("metrics", {})
        fm.trained_at = data.get("trained_at")
        fm.version = data.get("version", 1)
        return fm


# ═════════════════════════════════════════════════════════════════════
# ML ENGINE — Orchestratore principale
# ═════════════════════════════════════════════════════════════════════

class MLEngine:
    """Motore Machine Learning principale.

    Il modello impara schemi dai dati:
    - Ogni documento processato arricchisce il dataset
    - Ogni correzione umana migliora la qualità dei dati
    - Training produce modelli che estraggono automaticamente
    - Qualità dati → Qualità modello → Qualità estrazione

    Gestisce:
    - Raccolta automatica dati da ogni documento
    - Training supervisionato per campo
    - Inference con confidenza calibrata
    - Monitoring qualità dati e modelli
    - Raccomandazioni per migliorare le prestazioni
    """

    def __init__(self):
        self.data = DataStore()
        self.models: Dict[str, FieldModel] = {}
        self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
        self._load_models()
        self._migrate_old_data()

    def _migrate_old_data(self):
        """Migrazione una-tantum dalla vecchia tabella training_samples."""
        try:
            migrated = self.data.migrate_from_old_tables()
            if migrated > 0:
                logger.info(f"Migrati {migrated} esempi dalla vecchia tabella")
        except Exception:
            pass

    def _load_models(self):
        """Carica tutti i modelli addestrati dal disco."""
        # Nuovi modelli (formato mlmodel_*)
        for mf in MODEL_DIR.glob("mlmodel_*.pkl"):
            field = mf.stem.replace("mlmodel_", "")
            try:
                fm = FieldModel.load(mf)
                self.models[field] = fm
                logger.info(
                    f"Caricato modello ML: {field} "
                    f"(v{fm.version}, {fm.n_samples} campioni, "
                    f"acc={fm.metrics.get('accuracy', 'N/A')})"
                )
            except Exception as e:
                logger.warning(f"Errore caricamento {mf}: {e}")

        # Vecchi modelli (formato model_*) — importazione compatibilità
        for mf in MODEL_DIR.glob("model_*.pkl"):
            field = mf.stem.replace("model_", "")
            if field not in self.models:
                try:
                    with open(mf, "rb") as f:
                        old_model = pickle.load(f)
                    fm = FieldModel(field)
                    fm.model = old_model
                    fm.version = 1
                    fm.trained_at = datetime.now().isoformat()
                    self.models[field] = fm
                    logger.info(f"Importato vecchio modello: {field}")
                except Exception:
                    pass

    # ═══════════════════════════════════════════════════════════════
    # RACCOLTA DATI — ogni documento arricchisce il dataset
    # ═══════════════════════════════════════════════════════════════

    def collect_from_extraction(self, doc_text: str, extracted: dict,
                                methods: dict, doc_id: str = ""):
        """Auto-raccolta dati da un documento processato.

        Ogni campo estratto con successo diventa un esempio di training.
        La qualità dell'esempio dipende dal metodo di estrazione.
        """
        from field_registry import registry, get_validator

        validated_fields = set()
        for field, value in extracted.items():
            if field.startswith('_') or value is None:
                continue
            fd = registry.get(field)
            if fd and fd.validator_type:
                validator = get_validator(fd.validator_type)
                if validator:
                    try:
                        if validator(value):
                            validated_fields.add(field)
                    except Exception:
                        pass

        count = self.data.add_document_examples(
            doc_text, extracted, methods,
            doc_id=doc_id, validated_fields=validated_fields
        )
        return count

    def add_correction(self, field: str, text: str, correct_value: str,
                       wrong_value: str = None, doc_id: str = "") -> dict:
        """Registra una correzione umana — la fonte dati più preziosa.

        Le correzioni umane hanno la qualità più alta (1.0) e hanno
        il peso maggiore nel training. Più correzioni = modello migliore.
        """
        self.data.add_correction(
            field, text, correct_value,
            wrong_value=wrong_value, doc_id=doc_id
        )
        count = len(self.data.get_training_data(field))
        can_train = count >= MIN_SAMPLES_TRAIN
        return {
            "status": "ok",
            "field": field,
            "total_samples": count,
            "can_train": can_train,
            "message": (
                f"Correzione salvata nel dataset ML. "
                f"{count} campioni per '{field}'."
                + (f" Puoi addestrare il modello!" if can_train
                   else f" Servono ancora {MIN_SAMPLES_TRAIN - count} per addestrare.")
            ),
        }

    # ═══════════════════════════════════════════════════════════════
    # TRAINING — il modello impara dai dati
    # ═══════════════════════════════════════════════════════════════

    def train_field(self, field: str, min_quality: float = 0.0) -> dict:
        """Addestra/riaddestra il modello per un campo specifico.

        La qualità dei dati determina la qualità del modello:
        - Dati di alta qualità (correzioni) pesano di più
        - Cross-validation per metriche realistiche
        - Confronto con modello precedente → rollback se peggiora
        - Versionamento automatico
        """
        t_start = time.time()

        # Recupera dati
        data = self.data.get_training_data(field, min_quality=min_quality)
        if len(data) < MIN_SAMPLES_TRAIN:
            return {
                "status": "error",
                "field": field,
                "message": (
                    f"Dati insufficienti per '{field}': {len(data)}/{MIN_SAMPLES_TRAIN}. "
                    f"Processa più documenti o aggiungi correzioni."
                ),
                "n_samples": len(data),
            }

        texts = [d[0] for d in data]
        values = [d[1] for d in data]
        qualities = [d[2] for d in data]

        # Addestra nuovo modello
        new_model = FieldModel(field)
        result = new_model.train(texts, values, qualities)

        if result["status"] != "ok":
            return result

        # Confronta con modello vecchio
        old_model = self.models.get(field)
        old_metrics = {}
        should_replace = True

        if old_model and old_model.model and len(data) >= MIN_SAMPLES_CV:
            try:
                old_preds = [old_model.predict(t)[0] for t in texts]
                old_acc = sum(
                    1 for p, v in zip(old_preds, values) if p == v
                ) / len(values)
                old_metrics["accuracy"] = old_acc

                new_acc = result["metrics"]["accuracy"]
                if new_acc < old_acc + MIN_IMPROVEMENT:
                    should_replace = False
            except Exception:
                pass

        if not should_replace:
            return {
                "status": "rollback",
                "field": field,
                "message": (
                    f"⚠ Nuovo modello '{field}' non migliore del precedente. "
                    f"Nuovo: {result['metrics']['accuracy']:.1%} vs "
                    f"Vecchio: {old_metrics['accuracy']:.1%}. Mantenuto vecchio."
                ),
                "new_metrics": result["metrics"],
                "old_metrics": old_metrics,
            }

        # Salva e deploya
        model_path = MODEL_DIR / f"mlmodel_{field}.pkl"
        backup_path = MODEL_DIR / f"mlmodel_{field}_prev.pkl"

        if model_path.exists():
            shutil.copy2(model_path, backup_path)

        new_model.save(model_path)
        self.models[field] = new_model

        # Registra versione nel DB
        self._register_model_version(field, new_model)

        total_time = round((time.time() - t_start) * 1000, 1)
        avg_quality = round(sum(qualities) / len(qualities), 3)

        msg = (
            f"✅ Modello '{field}' v{new_model.version} addestrato: "
            f"{len(data)} campioni (qualità media: {avg_quality:.0%}), "
            f"accuracy {result['metrics']['accuracy']:.1%}"
        )
        if "cv_mean" in result["metrics"]:
            msg += (
                f", CV: {result['metrics']['cv_mean']:.1%}"
                f"±{result['metrics']['cv_std']:.1%}"
            )
        if old_metrics:
            msg += f" [prima: {old_metrics['accuracy']:.1%}]"
        msg += f" ({total_time}ms)"

        return {
            "status": "ok",
            "field": field,
            "message": msg,
            "version": new_model.version,
            "metrics": result["metrics"],
            "old_metrics": old_metrics,
            "training_time_ms": total_time,
            "n_samples": len(data),
            "data_quality_avg": avg_quality,
        }

    def train_all(self, min_quality: float = 0.0) -> dict:
        """Addestra modelli per TUTTI i campi con dati sufficienti.

        Il sistema identifica automaticamente quali campi hanno
        abbastanza dati e addestra un modello per ciascuno.
        """
        fields = self.data.get_all_fields()
        results = {}
        trained = 0
        errors = 0

        for field, count in fields.items():
            if count >= MIN_SAMPLES_TRAIN:
                result = self.train_field(field, min_quality=min_quality)
                results[field] = result
                if result["status"] == "ok":
                    trained += 1
                elif result["status"] == "error":
                    errors += 1

        return {
            "status": "ok",
            "trained": trained,
            "errors": errors,
            "total_fields": len(fields),
            "trainable_fields": sum(1 for c in fields.values() if c >= MIN_SAMPLES_TRAIN),
            "results": results,
            "message": (
                f"Addestrati {trained}/{len(fields)} modelli. "
                f"{sum(fields.values())} campioni totali nel dataset."
            ),
        }

    def rollback_field(self, field: str) -> dict:
        """Rollback al modello precedente."""
        backup = MODEL_DIR / f"mlmodel_{field}_prev.pkl"
        model_path = MODEL_DIR / f"mlmodel_{field}.pkl"

        if not backup.exists():
            return {"status": "error", "message": f"Nessun backup per '{field}'"}

        shutil.copy2(backup, model_path)
        self.models[field] = FieldModel.load(model_path)

        return {
            "status": "ok",
            "message": f"Rollback '{field}' completato. Modello precedente ripristinato.",
        }

    # ═══════════════════════════════════════════════════════════════
    # INFERENCE — il modello fa previsioni
    # ═══════════════════════════════════════════════════════════════

    def predict_field(self, field: str, text: str) -> Tuple[Optional[str], float, str]:
        """Predice un singolo campo. Ritorna (valore, confidenza, metodo)."""
        model = self.models.get(field)
        if model and model.model:
            value, confidence = model.predict(text)
            if value and confidence >= self.confidence_threshold:
                return value, confidence, "ml"
        return None, 0.0, "none"

    def predict_all(self, text: str) -> Dict[str, Tuple[str, float]]:
        """Predice tutti i campi con modelli ML.
        Ritorna {campo: (valore, confidenza)} per campi con confidenza sufficiente."""
        predictions = {}
        for field, model in self.models.items():
            if model and model.model:
                value, confidence = model.predict(text)
                if value and confidence >= self.confidence_threshold:
                    predictions[field] = (value, confidence)
        return predictions

    def fill_missing(self, result: dict, text: str) -> Tuple[dict, dict]:
        """Riempie campi vuoti usando modelli ML.
        Ritorna (risultato_aggiornato, metodi_ml)."""
        ml_methods = {}
        for field, model in self.models.items():
            if field.startswith('_'):
                continue
            current = result.get(field)
            if current in [None, "", 0]:
                value, confidence = model.predict(text[:5000])
                if value and confidence >= self.confidence_threshold:
                    result[field] = value
                    ml_methods[field] = f"ml({confidence:.0%})"
        return result, ml_methods

    def enhance_result(self, result: dict, text: str) -> Tuple[dict, dict]:
        """Migliora i risultati di estrazione con ML.

        - Campi vuoti: ML li riempie
        - Campi esistenti con bassa confidenza: ML può sovrascrivere
          se il modello è molto sicuro (>85%)

        Questo è il cuore dell'integrazione ML nel pipeline:
        le regole estraggono, il ML migliora.
        """
        ml_methods = {}

        for field, model in self.models.items():
            if field.startswith('_'):
                continue

            value, confidence = model.predict(text[:5000])
            if not value or confidence < self.confidence_threshold:
                continue

            current = result.get(field)

            if current in [None, "", 0]:
                # Campo vuoto → ML lo riempie
                result[field] = value
                ml_methods[field] = f"ml({confidence:.0%})"
            elif confidence > ML_OVERRIDE_THRESHOLD and str(current) != str(value):
                # ML molto sicuro e in disaccordo → sovrascrive
                result[field] = value
                ml_methods[field] = f"ml_override({confidence:.0%})"

        return result, ml_methods

    # ═══════════════════════════════════════════════════════════════
    # MONITORING — qualità dati e modelli
    # ═══════════════════════════════════════════════════════════════

    def get_status(self) -> dict:
        """Stato completo del motore ML."""
        data_stats = self.data.get_all_fields()

        models_info = {}
        for field, model in self.models.items():
            models_info[field] = {
                "version": model.version,
                "n_samples": model.n_samples,
                "accuracy": model.metrics.get("accuracy"),
                "f1": model.metrics.get("f1"),
                "cv_mean": model.metrics.get("cv_mean"),
                "cv_std": model.metrics.get("cv_std"),
                "n_classes": model.metrics.get("n_classes"),
                "trained_at": model.trained_at,
            }

        return {
            "engine": "MLEngine v1.0",
            "active_models": len(self.models),
            "total_training_data": sum(data_stats.values()) if data_stats else 0,
            "trainable_fields": sum(
                1 for c in data_stats.values() if c >= MIN_SAMPLES_TRAIN
            ),
            "data_per_field": data_stats,
            "models": models_info,
            "confidence_threshold": self.confidence_threshold,
            "min_samples_train": MIN_SAMPLES_TRAIN,
        }

    def get_quality_report(self) -> dict:
        """Report completo qualità dati e modelli.

        Mostra la relazione tra qualità dati → qualità modello.
        Include raccomandazioni per migliorare le prestazioni.
        """
        data_quality = self.data.get_data_quality()

        model_quality = {}
        for field, model in self.models.items():
            field_data = data_quality["by_field"].get(field, {})
            model_quality[field] = {
                "version": model.version,
                "metrics": model.metrics,
                "trained_at": model.trained_at,
                "data_samples": field_data.get("count", 0),
                "data_quality_avg": field_data.get("avg_quality", 0),
                "value_diversity": data_quality["value_diversity"].get(field, 0),
            }

        # Campi con dati ma senza modello
        untrained = {}
        for field, info in data_quality["by_field"].items():
            if field not in self.models:
                untrained[field] = {
                    "samples": info["count"],
                    "avg_quality": info["avg_quality"],
                    "can_train": info["count"] >= MIN_SAMPLES_TRAIN,
                }

        return {
            "data_quality": data_quality,
            "model_quality": model_quality,
            "untrained_fields": untrained,
            "recommendations": self._get_recommendations(data_quality, model_quality),
        }

    def _get_recommendations(self, data_quality, model_quality) -> List[str]:
        """Genera raccomandazioni per migliorare la qualità del modello.

        Principio: la qualità dei dati determina la qualità del modello.
        """
        recs = []

        total = data_quality["total_examples"]
        if total == 0:
            recs.append(
                "📊 Nessun dato di training. Processa documenti PDF per "
                "iniziare a raccogliere dati automaticamente."
            )
            return recs

        if total < 20:
            recs.append(
                "📊 Dataset piccolo. Processa più documenti per "
                "accumulare dati di training."
            )

        correction_count = data_quality["by_source"].get("correction", {}).get("count", 0)
        if correction_count < 5:
            recs.append(
                "✏️ Le correzioni umane sono i dati migliori (qualità 100%). "
                "Correggi errori di estrazione per migliorare drasticamente i modelli."
            )

        for field, info in data_quality["by_field"].items():
            if info["count"] >= MIN_SAMPLES_TRAIN and field not in model_quality:
                recs.append(
                    f"🔧 Campo '{field}' ha {info['count']} campioni. "
                    f"Addestralo con POST /api/ml/train/{field}"
                )

            if info["avg_quality"] < 0.4 and info["count"] >= 3:
                recs.append(
                    f"⚠️ Qualità dati bassa per '{field}' "
                    f"(media: {info['avg_quality']:.0%}). "
                    f"Aggiungi correzioni manuali per alzarla."
                )

        for field, info in model_quality.items():
            acc = info.get("metrics", {}).get("accuracy", 0)
            if acc and acc < 0.7:
                recs.append(
                    f"📈 Modello '{field}' ha accuracy {acc:.0%}. "
                    f"Più dati di alta qualità lo miglioreranno."
                )

        if not recs:
            recs.append(
                "✅ Il sistema sta imparando bene! Continua a processare "
                "documenti e fare correzioni per migliorare."
            )

        return recs

    def get_learning_curve(self, field: str) -> dict:
        """Curva di apprendimento: come l'accuracy cambia con più dati.

        Mostra graficamente la relazione:
        più dati → migliore accuracy (fino a saturazione).
        """
        data = self.data.get_training_data(field)
        if len(data) < MIN_SAMPLES_CV:
            return {
                "status": "error",
                "message": f"Dati insufficienti per curva di apprendimento ({len(data)} campioni)"
            }

        texts = [d[0] for d in data]
        values = [d[1] for d in data]

        if len(set(values)) < 2:
            return {
                "status": "error",
                "message": "Servono almeno 2 valori diversi per la curva"
            }

        sizes = []
        accuracies = []

        steps = min(10, len(data) - MIN_SAMPLES_TRAIN + 1)
        for i in range(steps):
            n = MIN_SAMPLES_TRAIN + int(
                i * (len(data) - MIN_SAMPLES_TRAIN) / max(steps - 1, 1)
            )
            n = min(n, len(data))

            t_sub = texts[:n]
            v_sub = values[:n]

            if len(set(v_sub)) < 2:
                continue

            try:
                model = SkPipeline([
                    ("features", FeatureUnion([
                        ("word", TfidfVectorizer(
                            analyzer="word", ngram_range=(1, 2),
                            max_features=5000, min_df=1, max_df=1.0
                        )),
                        ("char", TfidfVectorizer(
                            analyzer="char_wb", ngram_range=(2, 4),
                            max_features=5000, min_df=1, max_df=1.0
                        )),
                    ])),
                    ("clf", LogisticRegression(
                        C=1.0, max_iter=1000, solver='lbfgs',
                        class_weight='balanced'
                    )),
                ])

                min_class = min(Counter(v_sub).values())
                n_splits = min(3, min_class)
                if n_splits >= 2:
                    scores = cross_val_score(
                        model, t_sub, v_sub,
                        cv=StratifiedKFold(
                            n_splits=n_splits, shuffle=True, random_state=42
                        ),
                        scoring='accuracy'
                    )
                    sizes.append(n)
                    accuracies.append(float(np.mean(scores)))
            except Exception:
                continue

        return {
            "status": "ok",
            "field": field,
            "sizes": sizes,
            "accuracies": accuracies,
            "total_data": len(data),
            "message": (
                f"Curva di apprendimento per '{field}': "
                f"{len(sizes)} punti calcolati su {len(data)} campioni totali."
            ),
        }

    # ── DB Registration ────────────────────────────────────────────

    def _register_model_version(self, field: str, model: FieldModel):
        """Registra una versione del modello nel database."""
        with get_connection() as conn:
            c = conn.cursor()

            # Disattiva vecchie versioni
            c.execute(
                "UPDATE ml_model_registry SET is_active=0 WHERE field=?", (field,)
            )

            # Prossima versione
            current_v = c.execute(
                "SELECT MAX(version) FROM ml_model_registry WHERE field=?", (field,)
            ).fetchone()[0] or 0
            new_v = current_v + 1
            model.version = new_v

            c.execute(
                "INSERT INTO ml_model_registry "
                "(field, version, accuracy, f1_score, cv_score_mean, cv_score_std, "
                "n_samples, n_classes, training_time_ms, is_active, notes) "
                "VALUES (?,?,?,?,?,?,?,?,?,1,?)",
                (
                    field, new_v,
                    model.metrics.get("accuracy"),
                    model.metrics.get("f1"),
                    model.metrics.get("cv_mean"),
                    model.metrics.get("cv_std"),
                    model.n_samples,
                    model.metrics.get("n_classes"),
                    model.metrics.get("training_time_ms"),
                    json.dumps(model.metrics, default=str),
                )
            )

    def get_model_versions(self, field: str = None) -> list:
        """Storico versioni modelli."""
        with get_connection(readonly=True) as conn:
            c = conn.cursor()

            if field:
                rows = c.execute(
                    "SELECT field, version, accuracy, f1_score, cv_score_mean, "
                    "cv_score_std, n_samples, n_classes, training_time_ms, "
                    "is_active, notes, created_at "
                    "FROM ml_model_registry WHERE field=? ORDER BY version DESC",
                    (field,)
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT field, version, accuracy, f1_score, cv_score_mean, "
                    "cv_score_std, n_samples, n_classes, training_time_ms, "
                    "is_active, notes, created_at "
                    "FROM ml_model_registry ORDER BY created_at DESC"
                ).fetchall()

        return [{
            "field": r[0], "version": r[1], "accuracy": r[2], "f1": r[3],
            "cv_mean": r[4], "cv_std": r[5], "n_samples": r[6],
            "n_classes": r[7], "training_time_ms": r[8],
            "is_active": bool(r[9]), "notes": r[10], "created_at": r[11],
        } for r in rows]


# ═════════════════════════════════════════════════════════════════════
# SINGLETON — istanza globale
# ═════════════════════════════════════════════════════════════════════

ml_engine = MLEngine()
