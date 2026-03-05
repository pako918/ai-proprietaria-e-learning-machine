"""
AppaltoAI — Configurazione Centralizzata
Tutte le costanti, percorsi e soglie in un unico modulo.
Supporta override via variabili d'ambiente.
"""

import os
from pathlib import Path

# ═════════════════════════════════════════════════════════════════════════════
# PERCORSI
# ═════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "learning.db"

# Crea directory necessarie
for _dir in (DATA_DIR, MODEL_DIR, UPLOAD_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# SERVER
# ═════════════════════════════════════════════════════════════════════════════

SERVER_HOST = os.getenv("APPALTOAI_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("APPALTOAI_PORT", "8000"))
CORS_ORIGINS = os.getenv("APPALTOAI_CORS_ORIGINS", "http://localhost:8000").split(",")
API_VERSION = "3.0.0"
APP_TITLE = "AppaltoAI API"

# ═════════════════════════════════════════════════════════════════════════════
# UPLOAD
# ═════════════════════════════════════════════════════════════════════════════

MAX_UPLOAD_SIZE_MB = int(os.getenv("APPALTOAI_MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
MIN_TEXT_LENGTH = 20

# ═════════════════════════════════════════════════════════════════════════════
# ML ENGINE
# ═════════════════════════════════════════════════════════════════════════════

MIN_SAMPLES_TRAIN = int(os.getenv("APPALTOAI_MIN_SAMPLES", "5"))
MIN_SAMPLES_CV = 6
MIN_IMPROVEMENT = 0.0  # Non accettare modelli peggiori
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("APPALTOAI_CONFIDENCE_THRESHOLD", "0.30"))
ML_OVERRIDE_THRESHOLD = 0.85

DATA_QUALITY_WEIGHTS = {
    "correction":       1.0,
    "manual":           0.95,
    "approved":         0.90,
    "rules_validated":  0.60,
    "rules_raw":        0.35,
    "auto":             0.20,
}

# ═════════════════════════════════════════════════════════════════════════════
# QUARANTINE (legacy, supervised review)
# ═════════════════════════════════════════════════════════════════════════════

QUARANTINE_THRESHOLD = float(os.getenv("APPALTOAI_QUARANTINE_THRESHOLD", "0.50"))
QUARANTINE_BLACKLIST = frozenset({
    "note_operative", "lotti", "criteri_tecnici", "vincoli_lotti",
    "scadenze", "struttura_compenso_65_35",
    "categorie_ingegneria", "sopralluogo_note",
})

# ═════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═════════════════════════════════════════════════════════════════════════════

LOG_LEVEL = os.getenv("APPALTOAI_LOG_LEVEL", "INFO")
