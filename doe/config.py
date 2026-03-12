"""Configurazione DOE (Directive-Orchestration-Execution)."""

import os
from pathlib import Path

# ── Directory ─────────────────────────────────────────────────────
DOE_DIR = Path(__file__).parent
DIRECTIVES_DIR = DOE_DIR / "directives"
LEARNED_DIR = DIRECTIVES_DIR / "learned"

# ── LLM locale (Ollama) ──────────────────────────────────────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# ── Soglie orchestratore ─────────────────────────────────────────
REFINE_CONFIDENCE_THRESHOLD = float(
    os.getenv("DOE_REFINE_THRESHOLD", "0.5")
)
MAX_AGENT_STEPS = int(os.getenv("DOE_MAX_STEPS", "10"))
MAX_RETRIES_PER_FIELD = 2

# ── Self-learning ────────────────────────────────────────────────
EVOLUTION_MIN_CORRECTIONS = int(
    os.getenv("DOE_EVOLUTION_MIN_CORRECTIONS", "3")
)

# Campi troppo strutturati per un LLM 7B — restano solo deterministici
SKIP_LLM_FIELDS = frozenset({
    "lotti", "criteri_valutazione", "sub_criteri",
    "categorie_professionisti", "criteri_tecnici",
})

# ── Init ──────────────────────────────────────────────────────────
LEARNED_DIR.mkdir(parents=True, exist_ok=True)
