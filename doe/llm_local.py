"""Client per LLM locale via Ollama — zero dipendenze esterne.

Usa l'API HTTP di Ollama (http://localhost:11434) con urllib della stdlib.
Supporta generazione testo, chat multi-turno e output JSON strutturato.
"""

import json
import logging
import urllib.request
import urllib.error

from .config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT

log = logging.getLogger(__name__)


class OllamaClient:
    """Client HTTP per comunicare con Ollama in locale."""

    def __init__(self, host: str | None = None, model: str | None = None):
        self.host = (host or OLLAMA_HOST).rstrip("/")
        self.model = model or OLLAMA_MODEL

    # ── Disponibilità ─────────────────────────────────────────────

    def is_available(self) -> bool:
        """Verifica se Ollama è in esecuzione."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def list_models(self) -> list[str]:
        """Elenca i modelli installati su Ollama."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                return [m["name"] for m in data.get("models", [])]
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return []

    # ── Generazione testo ─────────────────────────────────────────

    def generate(self, prompt: str, system: str | None = None,
                 temperature: float = 0.1, max_tokens: int = 4096) -> str:
        """Generazione di testo semplice (singolo turno)."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system
        return self._post("/api/generate", payload).get("response", "")

    def chat(self, messages: list[dict], temperature: float = 0.1,
             max_tokens: int = 4096) -> str:
        """Chat completion multi-turno."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        resp = self._post("/api/chat", payload)
        return resp.get("message", {}).get("content", "")

    def generate_json(self, prompt: str, system: str | None = None,
                      temperature: float = 0.0) -> dict | None:
        """Generazione con output JSON forzato (format: json)."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temperature,
                "num_predict": 4096,
            },
        }
        if system:
            payload["system"] = system
        raw = self._post("/api/generate", payload).get("response", "")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            log.warning("LLM non ha restituito JSON valido: %s", raw[:200])
            return None

    # ── HTTP interno ──────────────────────────────────────────────

    def _post(self, path: str, payload: dict) -> dict:
        """Chiamata POST all'API Ollama."""
        url = f"{self.host}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            log.error("Ollama HTTP %d: %s", e.code, body[:300])
            raise
        except urllib.error.URLError as e:
            log.error("Ollama non raggiungibile: %s", e.reason)
            raise ConnectionError(
                f"Ollama non raggiungibile su {self.host}. "
                "Avvia Ollama con: ollama serve"
            ) from e

    def __repr__(self):
        return f"OllamaClient(host={self.host!r}, model={self.model!r})"
