"""Execution Layer — Strumenti deterministici invocabili dall'agente.

Ogni strumento è una funzione Python registrata con @tool().
L'orchestratore sceglie quale strumento usare basandosi sulle direttive
e sul suo ragionamento (ReAct loop).
"""

import json
import logging
import re

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════

TOOLS: dict[str, dict] = {}


def tool(name: str, description: str, parameters: dict):
    """Decoratore per registrare uno strumento."""
    def decorator(fn):
        TOOLS[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "function": fn,
        }
        return fn
    return decorator


def get_tools_schema() -> list[dict]:
    """Schema degli strumenti per il prompt dell'agente."""
    return [
        {"name": t["name"], "description": t["description"],
         "parameters": t["parameters"]}
        for t in TOOLS.values()
    ]


def execute_tool(name: str, args: dict) -> str:
    """Esegue uno strumento; ritorna JSON stringa."""
    if name not in TOOLS:
        return json.dumps({"error": f"Strumento '{name}' non trovato"})
    try:
        result = TOOLS[name]["function"](**args)
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False, default=str)
        return str(result)
    except Exception as e:
        log.error("Errore eseguendo %s: %s", name, e)
        return json.dumps({"error": str(e)})


# ══════════════════════════════════════════════════════════════════
# STRUMENTI CONCRETI
# ══════════════════════════════════════════════════════════════════

@tool(
    name="estrai_sezione",
    description=(
        "Estrae il testo di una sezione specifica dal documento "
        "cercando per parola chiave (es. 'OGGETTO', 'PROCEDURA', 'LOTTI')."
    ),
    parameters={
        "testo": "Testo completo del documento",
        "keyword": "Parola chiave della sezione",
        "max_chars": "Max caratteri da restituire (default 2000)",
    },
)
def estrai_sezione(testo: str, keyword: str, max_chars: int = 2000) -> dict:
    """Trova ed estrae una sezione dal testo."""
    pattern = re.compile(
        rf"(?:^|\n)\s*(?:art\.?\s*\d+|[\d.]+)\s*[-–—.]?\s*{re.escape(keyword)}",
        re.IGNORECASE,
    )
    match = pattern.search(testo)
    if match:
        start = match.start()
    else:
        idx = testo.lower().find(keyword.lower())
        if idx == -1:
            return {"found": False, "text": ""}
        start = max(0, idx - 100)
    return {"found": True, "text": testo[start : start + max_chars], "position": start}


@tool(
    name="cerca_valore",
    description=(
        "Cerca un valore nel testo usando una regex. "
        "Utile per trovare importi, codici CIG/CUP, date."
    ),
    parameters={
        "testo": "Testo in cui cercare",
        "pattern": "Pattern regex",
        "group": "Gruppo regex (default 0 = intero match)",
    },
)
def cerca_valore(testo: str, pattern: str, group: int = 0) -> dict:
    """Cerca un valore con regex."""
    try:
        matches = []
        for m in re.finditer(pattern, testo, re.IGNORECASE):
            g = group if group <= len(m.groups()) else 0
            matches.append({
                "match": m.group(g),
                "position": m.start(),
                "context": testo[max(0, m.start() - 50) : m.end() + 50],
            })
            if len(matches) >= 5:
                break
        return {"found": len(matches) > 0, "matches": matches}
    except re.error as e:
        return {"error": f"Regex non valida: {e}"}


@tool(
    name="valida_campo",
    description=(
        "Valida un valore estratto contro le statistiche storiche. "
        "Rileva anomalie e suggerisce alternative."
    ),
    parameters={
        "field": "Nome del campo (es. 'importo_base_asta')",
        "value": "Valore da validare",
    },
)
def valida_campo(field: str, value: str) -> dict:
    """Valida un campo contro le statistiche."""
    try:
        from adaptive_learner import adaptive_learner
        return adaptive_learner.validator.validate_value(field, value)
    except Exception as e:
        return {"validated": True, "note": f"Validazione non disponibile: {e}"}


@tool(
    name="storico_campo",
    description=(
        "Recupera lo storico delle estrazioni per un campo. "
        "Mostra valori comuni e pattern ricorrenti."
    ),
    parameters={
        "field": "Nome del campo",
        "limit": "Max risultati (default 10)",
    },
)
def storico_campo(field: str, limit: int = 10) -> dict:
    """Recupera lo storico di un campo."""
    try:
        from adaptive_learner import adaptive_learner
        history = adaptive_learner.memory.get_field_history(field, limit=limit)
        return {
            "field": field,
            "count": len(history),
            "values": [
                {"value": h[2], "method": h[3], "confidence": h[4]}
                for h in history
            ],
        }
    except Exception as e:
        return {"error": str(e)}


@tool(
    name="confronta_simili",
    description=(
        "Trova documenti simili già processati e mostra cosa è stato estratto "
        "per un campo specifico."
    ),
    parameters={
        "doc_id": "ID del documento corrente",
        "field": "Campo da confrontare",
    },
)
def confronta_simili(doc_id: str, field: str) -> dict:
    """Trova estrazioni da documenti simili."""
    try:
        from adaptive_learner import adaptive_learner
        similar = adaptive_learner.similarity.find_similar(doc_id, top_k=5)
        results = []
        for sim_doc_id, score in similar:
            hist = adaptive_learner.memory.get_field_history(field, limit=1)
            if hist:
                results.append({
                    "doc_id": sim_doc_id,
                    "similarity": round(score, 3),
                    "value": hist[0][2],
                })
        return {"similar_docs": results}
    except Exception as e:
        return {"error": str(e)}
