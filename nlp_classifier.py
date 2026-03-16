"""
AppaltoAI — Classificatore NLP
Classificatori ML specializzati per tipologia procedura e criteri.
Il modello NON genera testo — solo classificazione e entity recognition.
"""

import pickle
import re
from typing import Optional, Dict, Tuple, Any

from config import MODEL_DIR
from utils import clean_string
from log_config import get_logger

logger = get_logger("nlp_classifier")


class NLPClassifier:
    """Classificatori ML specializzati. Il modello NON genera testo."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self._load_models()

    def _load_models(self):
        for mf in MODEL_DIR.glob("model_*.pkl"):
            field = mf.stem.replace("model_", "")
            try:
                with open(mf, "rb") as f:
                    self.models[field] = pickle.load(f)
            except Exception:
                pass

    def reload(self):
        self.models.clear()
        self._load_models()

    def classify_procedure(self, text: str) -> str:
        if re.search(r'accordo\s+quadro', text, re.I):
            tipo = "a unico operatore" if re.search(r'unico\s+operatore', text, re.I) else ""
            n = re.search(r'suddivisa?\s+in\s+(\d+|due|tre)\s+lott', text, re.I)
            lotti = f" — {n.group(1)} Lotti" if n else ""
            return f"Procedura Aperta Telematica — Accordo Quadro {tipo}{lotti}".strip()
        if re.search(r'procedura\s+(?:telematica\s+)?aperta', text, re.I):
            n = re.search(r'suddivisa?\s+in\s+(\d+|due|tre)\s+lott', text, re.I)
            lotti = f" — {n.group(1)} Lotti" if n else ""
            return f"Procedura Aperta Telematica{lotti}"
        if re.search(r'procedura\s+ristretta', text, re.I):
            return "Procedura Ristretta"
        if re.search(r'procedura\s+negoziata', text, re.I):
            return "Procedura Negoziata"
        if re.search(r'affidamento\s+diretto', text, re.I):
            return "Affidamento Diretto"
        if re.search(r'(?:RdO|MEPA|mercato\s+elettronico)', text, re.I):
            return "RdO / MePA"
        pred = self._predict("tipo_procedura", text[:3000])
        return pred if pred else "Non specificata"

    def classify_criterio(self, text: str) -> str:
        if re.search(r'offerta\s+economicamente\s+piu\s+vantaggiosa|OEPV', text, re.I):
            art = re.search(r'art(?:icolo)?\.?\s*108\s+comma\s+\d+', text, re.I)
            suffix = f" ({clean_string(art.group(0))} D.Lgs. 36/2023)" if art else " (art. 108 D.Lgs. 36/2023)"
            return "OEPV — Offerta Economicamente Più Vantaggiosa" + suffix
        if re.search(r'(?:massimo|minor|minimo)\s+ribasso', text, re.I):
            return "Massimo/Minor Ribasso"
        if re.search(r'prezzo\s+piu\s+basso', text, re.I):
            return "Prezzo più basso"
        pred = self._predict("criterio_aggiudicazione", text[:3000])
        return pred if pred else "Non specificato"

    def fill_missing(self, result: dict, text: str) -> Tuple[dict, dict]:
        """ML fallback per campi vuoti. Ritorna (result, ml_methods)."""
        ml_methods = {}
        for key, value in result.items():
            if key.startswith('_') or isinstance(value, (bool, dict, list)):
                continue
            if value in [None, "", 0] and key in self.models:
                pred = self._predict(key, text[:3000])
                if pred:
                    result[key] = pred
                    ml_methods[key] = "ml"
        return result, ml_methods

    def _predict(self, field: str, snippet: str) -> Optional[str]:
        model = self.models.get(field)
        if model and snippet:
            try:
                return model.predict([snippet])[0]
            except Exception:
                return None
        return None
