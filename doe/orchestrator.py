"""Orchestration Layer — Agente AI con ciclo ReAct.

L'agente:
1. Legge le direttive (SOP)
2. Ragiona sul problema
3. Sceglie uno strumento o produce la risposta
4. Osserva il risultato
5. Ripete fino a completamento

Integrato nella pipeline come fase opzionale di raffinamento:
viene invocato SOLO per campi con bassa confidenza o vuoti,
dopo che l'estrazione deterministica ha già fatto il grosso del lavoro.
"""

import json
import logging
import re
from typing import Any

from .config import (
    MAX_AGENT_STEPS,
    REFINE_CONFIDENCE_THRESHOLD,
    SKIP_LLM_FIELDS,
)
from .llm_local import OllamaClient
from .directives import DirectiveManager
from .tools import get_tools_schema, execute_tool

log = logging.getLogger(__name__)

# ── Prompt di sistema dell'agente ─────────────────────────────────

AGENT_SYSTEM = """Sei un agente AI specializzato nell'analisi di disciplinari di gara italiani.

## Come Funzioni
Operi in un ciclo Pensa → Agisci → Osserva:
1. PENSA: Analizza il problema e decidi cosa fare
2. AGISCI: Usa uno strumento O dai la risposta finale
3. OSSERVA: Leggi il risultato e torna al passo 1

## Formato di Risposta
Rispondi SEMPRE in questo formato:

PENSIERO: [il tuo ragionamento]
AZIONE: nome_strumento
ARGOMENTI: {{"param1": "valore1"}}

OPPURE se hai finito:

PENSIERO: [ragionamento finale]
RISPOSTA: {{"value": "...", "confidence": 0.8, "motivazione": "..."}}

## Strumenti Disponibili
{tools}

## Direttive
{directives}

## Regole
- Non inventare MAI valori non presenti nel testo
- Preferisci estrazioni dal testo esatto del documento
- Se non trovi un valore, rispondi con null
- Importi come numeri senza € (es. 150000.00)
- Date in formato YYYY-MM-DD
- Fornisci SEMPRE una motivazione
"""


class DOEOrchestrator:
    """Agente AI che orchestra estrazione e raffinamento."""

    def __init__(self, llm: OllamaClient | None = None,
                 directives: DirectiveManager | None = None):
        self.llm = llm or OllamaClient()
        self.directives = directives or DirectiveManager()
        self._system_prompt: str | None = None

    # ── Prompt building ───────────────────────────────────────────

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = AGENT_SYSTEM.format(
                tools=json.dumps(get_tools_schema(), indent=2, ensure_ascii=False),
                directives=self.directives.get_system_prompt(),
            )
        return self._system_prompt

    def reload_directives(self):
        """Ricarica direttive e rigenera prompt."""
        self.directives.reload()
        self._system_prompt = None

    # ── Raffinamento (entry-point per la pipeline) ────────────────

    def refine_extraction(self, result: dict, text: str,
                          methods: dict | None = None,
                          doc_id: str | None = None) -> dict:
        """Raffina i campi con bassa confidenza usando l'LLM locale.

        Restituisce una copia di *result* con i campi migliorati
        e un sotto-dict ``_llm_improvements`` che elenca le modifiche.
        Se Ollama non è disponibile, restituisce *result* invariato.
        """
        if not self.llm.is_available():
            log.warning("Ollama non disponibile — skip raffinamento LLM")
            return result

        methods = methods or {}
        refined = dict(result)
        improvements: dict[str, dict] = {}

        fields_to_refine = self._find_weak_fields(result, methods)
        if not fields_to_refine:
            log.info("Nessun campo da raffinare con LLM")
            return result

        log.info("LLM: raffinamento di %d campi: %s",
                 len(fields_to_refine), list(fields_to_refine.keys()))

        for field, reason in fields_to_refine.items():
            if field in SKIP_LLM_FIELDS:
                continue
            improvement = self._refine_field(field, reason, text, result)
            if improvement and improvement.get("value") is not None:
                old_val = _get_nested(result, field)
                new_val = improvement["value"]
                if new_val != old_val:
                    _set_nested(refined, field, new_val)
                    improvements[field] = {
                        "old": old_val,
                        "new": new_val,
                        "confidence": improvement.get("confidence", 0.6),
                        "reason": improvement.get("motivazione", ""),
                    }
                    log.info("LLM migliorato %s: %s → %s",
                             field, repr(old_val)[:50], repr(new_val)[:50])

        if improvements:
            refined["_llm_improvements"] = improvements
            log.info("LLM ha migliorato %d campi", len(improvements))

        return refined

    # ── Analisi errore (utile per il self-learner) ────────────────

    def analyze_error(self, field: str, expected: str,
                      got: str, text: str) -> dict:
        """Analizza un errore di estrazione e suggerisce correzioni."""
        if not self.llm.is_available():
            return {"analysis": "LLM non disponibile"}

        prompt = (
            f'Analizza questo errore di estrazione da un disciplinare di gara:\n\n'
            f'Campo: {field}\n'
            f'Valore estratto (sbagliato): {got}\n'
            f'Valore corretto: {expected}\n\n'
            f'Testo del documento (estratto):\n{text[:4000]}\n\n'
            f'Rispondi in JSON:\n'
            f'{{"causa_errore": "...", "pattern_corretto": "...", '
            f'"regola_nuova": "...", "sezione_documento": "..."}}'
        )
        return self.llm.generate_json(prompt, system=self.system_prompt) or {}

    # ── Internals ─────────────────────────────────────────────────

    def _find_weak_fields(self, result: dict, methods: dict) -> dict[str, str]:
        """Identifica campi deboli da raffinare."""
        weak: dict[str, str] = {}
        for field, value in _flatten_dict(result):
            if field.startswith("_"):
                continue
            if value is None or value == "" or value == []:
                weak[field] = "campo_vuoto"
            elif field in methods:
                info = methods[field]
                if isinstance(info, dict):
                    conf = info.get("confidence", 1.0)
                    if conf < REFINE_CONFIDENCE_THRESHOLD:
                        weak[field] = f"bassa_confidenza ({conf:.2f})"
        return weak

    def _refine_field(self, field: str, reason: str,
                      text: str, current_result: dict) -> dict | None:
        """Ciclo ReAct per raffinare un singolo campo."""
        text_chunk = text[:8000]
        current_value = _get_nested(current_result, field)

        user_prompt = (
            f'Devo estrarre il campo "{field}" da un disciplinare di gara.\n\n'
            f'Stato attuale:\n'
            f'- Valore corrente: {json.dumps(current_value, ensure_ascii=False)}\n'
            f'- Problema: {reason}\n\n'
            f'Testo del documento (primi 8000 caratteri):\n---\n{text_chunk}\n---\n\n'
            f'Trova il valore corretto per "{field}". Se non lo trovi, rispondi null.'
        )

        messages = [{"role": "user", "content": user_prompt}]

        for step in range(MAX_AGENT_STEPS):
            response = self.llm.chat(
                [{"role": "system", "content": self.system_prompt}] + messages,
                temperature=0.1,
                max_tokens=2048,
            )
            if not response:
                break

            messages.append({"role": "assistant", "content": response})
            parsed = self._parse_response(response)

            if parsed["type"] == "answer":
                return parsed.get("data")

            if parsed["type"] == "tool_call":
                tool_name = parsed["tool"]
                tool_args = parsed["args"]
                if "testo" in tool_args and tool_args["testo"] == "DOCUMENTO":
                    tool_args["testo"] = text
                tool_result = execute_tool(tool_name, tool_args)
                messages.append({
                    "role": "user",
                    "content": f"Risultato di {tool_name}:\n{tool_result}",
                })
                continue

            # Forza risposta se sta per esaurire i passi
            if step >= MAX_AGENT_STEPS - 2:
                messages.append({
                    "role": "user",
                    "content": (
                        "Dai la tua RISPOSTA finale adesso. Formato: "
                        'RISPOSTA: {"value": ..., "confidence": ..., "motivazione": ...}'
                    ),
                })

        return None

    @staticmethod
    def _parse_response(text: str) -> dict:
        """Parse della risposta dell'agente (PENSIERO/AZIONE/RISPOSTA)."""
        # Cerca RISPOSTA finale
        resp_match = re.search(r"RISPOSTA\s*:\s*(\{.*?\})", text, re.DOTALL)
        if resp_match:
            try:
                return {"type": "answer", "data": json.loads(resp_match.group(1))}
            except json.JSONDecodeError:
                pass

        # Cerca AZIONE + ARGOMENTI
        action_match = re.search(r"AZIONE\s*:\s*(\w+)", text)
        if action_match:
            tool_name = action_match.group(1)
            args_match = re.search(r"ARGOMENTI\s*:\s*(\{.*?\})", text, re.DOTALL)
            tool_args = {}
            if args_match:
                try:
                    tool_args = json.loads(args_match.group(1))
                except json.JSONDecodeError:
                    pass
            return {"type": "tool_call", "tool": tool_name, "args": tool_args}

        # Prova JSON diretto con "value"
        json_match = re.search(r'\{[^{}]*"value"\s*:', text)
        if json_match:
            start = json_match.start()
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                if depth == 0:
                    try:
                        return {"type": "answer", "data": json.loads(text[start : i + 1])}
                    except json.JSONDecodeError:
                        break

        return {"type": "unknown", "raw": text}


# ── Utility per dict annidati ─────────────────────────────────────

def _flatten_dict(d: dict, prefix: str = "") -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and not k.startswith("_"):
            items.extend(_flatten_dict(v, path))
        else:
            items.append((path, v))
    return items


def _get_nested(d: dict, path: str):
    cur = d
    for k in path.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def _set_nested(d: dict, path: str, value):
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value
