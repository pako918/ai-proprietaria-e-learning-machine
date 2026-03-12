"""Directive Layer — Carica e gestisce le SOP in linguaggio naturale.

Le direttive sono file Markdown in doe/directives/:
  - sistema.md      → istruzioni di sistema (ruolo, strategia)
  - estrazione.md   → regole per-campo
  - validazione.md  → regole di validazione
  - learned/*.md    → direttive auto-generate dal self-learning
"""

import logging
from datetime import datetime
from pathlib import Path

from .config import DIRECTIVES_DIR, LEARNED_DIR

log = logging.getLogger(__name__)


class DirectiveManager:
    """Gestisce le direttive (SOP) in formato Markdown."""

    def __init__(self):
        self._cache: dict[str, str] = {}
        self._load_all()

    # ── Caricamento ───────────────────────────────────────────────

    def _load_all(self):
        """Carica tutte le direttive da disco."""
        self._cache.clear()
        for md in DIRECTIVES_DIR.glob("*.md"):
            self._cache[md.stem] = md.read_text(encoding="utf-8")
        for md in LEARNED_DIR.glob("*.md"):
            self._cache[f"learned/{md.stem}"] = md.read_text(encoding="utf-8")
        n_learned = sum(1 for k in self._cache if k.startswith("learned/"))
        log.info("Caricate %d direttive (%d learned)", len(self._cache), n_learned)

    def reload(self):
        """Ricarica tutte le direttive da disco."""
        self._load_all()

    # ── Accesso ───────────────────────────────────────────────────

    def get_system_prompt(self) -> str:
        """Assembla il prompt di sistema completo dalle direttive."""
        parts = []
        for key in ("sistema", "estrazione", "validazione"):
            if key in self._cache:
                parts.append(self._cache[key])
        # Append direttive apprese (hanno priorità sui default)
        for key in sorted(self._cache):
            if key.startswith("learned/"):
                parts.append(f"## Direttiva Appresa: {key}\n{self._cache[key]}")
        return "\n\n---\n\n".join(parts)

    def get_field_directive(self, field: str) -> str | None:
        """Ritorna la direttiva specifica per un campo, se esiste."""
        # Learned ha priorità
        learned_key = f"learned/{field}"
        if learned_key in self._cache:
            return self._cache[learned_key]
        # Cerca nel documento estrazione
        estrazione = self._cache.get("estrazione", "")
        marker = f"## {field}"
        if marker in estrazione:
            start = estrazione.index(marker)
            next_section = estrazione.find("\n## ", start + len(marker))
            return estrazione[start:] if next_section == -1 else estrazione[start:next_section]
        return None

    # ── Evoluzione ────────────────────────────────────────────────

    def save_learned_directive(self, field: str, content: str, reason: str):
        """Salva una nuova direttiva appresa dal self-learning."""
        header = (
            f"# Direttiva Appresa: {field}\n"
            f"<!-- Generata: {datetime.now().isoformat()} -->\n"
            f"<!-- Motivo: {reason} -->\n\n"
        )
        path = LEARNED_DIR / f"{field}.md"
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            full = existing + f"\n\n---\n\n{header}{content}"
        else:
            full = header + content
        path.write_text(full, encoding="utf-8")
        self._cache[f"learned/{field}"] = full
        log.info("Direttiva appresa salvata: %s", field)
