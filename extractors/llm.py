"""Integrazione LLM per estrazione (OpenAI / Anthropic)."""

import json
from pathlib import Path


PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompt_estrazione.json"


def load_prompt() -> dict:
    """Carica lo schema di estrazione dal file JSON."""
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_extraction_prompt(pdf_text: str) -> list[dict]:
    """Costruisce il prompt completo per l'LLM."""
    prompt_data = load_prompt()
    system_msg = prompt_data["system_prompt"]
    istruzioni = "\n".join(prompt_data["istruzioni_estrazione"])
    schema = json.dumps(prompt_data["schema_output"], indent=2, ensure_ascii=False)

    user_msg = f"""Analizza il seguente disciplinare di gara ed estrai TUTTI i dati strutturati.

ISTRUZIONI SPECIFICHE:
{istruzioni}

SCHEMA JSON DA COMPILARE (ogni campo ha una descrizione del valore atteso):
{schema}

REGOLE:
- Restituisci SOLO il JSON compilato, nessun testo aggiuntivo
- Usa null per i campi non trovati nel documento
- Per gli importi usa numeri senza simboli (es. 150000.00)
- Per le date usa formato YYYY-MM-DD dove possibile
- Estrai TUTTI i lotti, TUTTE le categorie, TUTTI i criteri

---
TESTO DEL DISCIPLINARE:
{pdf_text}
---

JSON compilato:"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def call_openai(messages: list[dict], model: str = "gpt-4o") -> str:
    """Chiama l'API OpenAI."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Installa openai: pip install openai")

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=16000,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def call_anthropic(messages: list[dict], model: str = "claude-sonnet-4-20250514") -> str:
    """Chiama l'API Anthropic."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("Installa anthropic: pip install anthropic")

    client = anthropic.Anthropic()
    system_msg = ""
    user_msgs = []
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            user_msgs.append(m)

    response = client.messages.create(
        model=model,
        max_tokens=16000,
        system=system_msg,
        messages=user_msgs,
        temperature=0.0,
    )
    return response.content[0].text
