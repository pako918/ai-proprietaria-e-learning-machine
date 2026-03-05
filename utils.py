"""
AppaltoAI — Utility condivise
Funzioni di supporto usate da più moduli: pulizia testo, parsing numeri,
ricerca contesto, normalizzazione importi.
"""

import re
from typing import Optional


def clean_string(s: Optional[str]) -> Optional[str]:
    """Pulisce e normalizza una stringa: collassa spazi, rimuove trailing punteggiatura."""
    if not s:
        return s
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'[,;:\s]+$', '', s)
    return s


def normalize_amount(raw) -> Optional[str]:
    """Normalizza un importo monetario nel formato italiano (€ 1.234,56)."""
    if not raw:
        return None
    raw = str(raw).strip().replace("EUR", "").replace("€", "").strip()
    if re.search(r'\d{1,3}(?:\.\d{3})+,\d{2}', raw):
        raw = raw.replace(".", "").replace(",", ".")
    elif re.search(r'\d{1,3}(?:,\d{3})+\.\d{2}', raw):
        raw = raw.replace(",", "")
    else:
        raw = raw.replace(",", ".")
    try:
        num = float(re.sub(r'[^\d.]', '', raw))
        return f"€ {num:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        return raw


def parse_number_word(s) -> Optional[int]:
    """Converte un numero o parola italiana (due, tre...) in intero."""
    if not s:
        return None
    mapping = {"due": 2, "tre": 3, "quattro": 4, "cinque": 5, "sei": 6}
    sl = s.strip().lower()
    if sl in mapping:
        return mapping[sl]
    try:
        return int(re.sub(r'\D', '', s))
    except (ValueError, TypeError):
        return None


def find_value_context(text: str, value: str, window: int = 300) -> str:
    """Trova il contesto testuale dove appare un valore nel documento.

    Cerca in ordine:
    1. Match diretto della stringa
    2. Match di singole parole significative (>3 caratteri)
    3. Match di sequenze numeriche (per importi)

    Ritorna il frammento di testo circostante per il training ML.
    """
    if not value or not text:
        return ""
    val_str = str(value).strip()
    if len(val_str) < 2:
        return ""

    # Ricerca diretta
    idx = text.lower().find(val_str.lower())
    if idx >= 0:
        start = max(0, idx - window)
        end = min(len(text), idx + len(val_str) + window)
        return text[start:end].strip()

    # Ricerca parole significative
    for word in (w for w in val_str.split() if len(w) > 3):
        idx = text.lower().find(word.lower())
        if idx >= 0:
            start = max(0, idx - window)
            end = min(len(text), idx + len(word) + window)
            return text[start:end].strip()

    # Ricerca numeri (per importi)
    for num in re.findall(r'\d{3,}', val_str):
        idx = text.find(num)
        if idx >= 0:
            start = max(0, idx - window)
            end = min(len(text), idx + len(num) + window)
            return text[start:end].strip()

    return ""


def first_match(text: str, patterns: list) -> Optional[str]:
    """Ritorna il primo match tra una lista di pattern regex."""
    for pat in patterns:
        try:
            m = re.search(pat, text, re.I | re.M)
            if m:
                return clean_string(m.group(1) if m.lastindex else m.group(0))
        except re.error:
            continue
    return None


def all_matches(text: str, patterns: list) -> list:
    """Ritorna tutti i match unici tra una lista di pattern regex."""
    found = []
    for pat in patterns:
        try:
            found.extend(re.findall(pat, text, re.I | re.M))
        except re.error:
            continue
    return list(dict.fromkeys([clean_string(x) for x in found if x]))


def extract_int(text: str, patterns: list) -> Optional[int]:
    """Estrae il primo intero trovato tramite pattern regex."""
    val = first_match(text, patterns)
    if val:
        try:
            return int(re.sub(r'\D', '', val))
        except (ValueError, TypeError):
            return None
    return None
