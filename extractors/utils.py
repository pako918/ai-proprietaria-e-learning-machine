"""Utilità condivise per l'estrazione da disciplinari."""

import re
from typing import Optional


def _parse_euro(raw: str) -> float | None:
    """Converte una stringa di importo in float. Gestisce i formati italiani."""
    if not raw:
        return None
    s = raw.strip().replace("€", "").replace("\u20ac", "").replace("Ç", "").strip()
    s = s.lstrip(".")
    s = s.rstrip(".,;: ")
    s = re.sub(r"\s+", "", s)
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    try:
        val = float(s)
        return val if val > 0 else None
    except (ValueError, TypeError):
        return None


def _find_all_euros(text_fragment: str) -> list[float]:
    """Trova tutti gli importi in euro in un frammento di testo."""
    raw = re.findall(
        r"(?:€|Ç|euro)\s*\.?\s*([\d.,]+)|"
        r"([\d]{1,3}(?:\.[\d]{3})*(?:,\d{1,2})?)\s*(?:€|Ç|euro|\bEur\b)",
        text_fragment,
        re.IGNORECASE,
    )
    results = []
    for groups in raw:
        for g in groups:
            v = _parse_euro(g)
            if v and v > 1:
                results.append(v)
    return results


def _section_text(full_text: str, start_patterns: list[str], end_patterns: list[str], max_len: int = 15000) -> str:
    """Estrae il testo di una sezione delimitata da heading iniziale e finale."""
    text_lower = full_text.lower()
    start_idx = -1

    def _is_toc_entry(pos: int) -> bool:
        after = full_text[pos:pos + 300]
        return bool(re.search(r"\.{5,}", after))

    for pat in start_patterns:
        pat_lower = pat.lower()
        search_from = 0
        while True:
            idx = text_lower.find(pat_lower, search_from)
            if idx < 0:
                break
            if not _is_toc_entry(idx):
                start_idx = idx
                break
            search_from = idx + len(pat_lower)
        if start_idx >= 0:
            break
        words = pat_lower.split()
        if len(words) >= 2:
            flex_pat = r"\s+".join(re.escape(w) for w in words)
            for m in re.finditer(flex_pat, text_lower):
                if not _is_toc_entry(m.start()):
                    start_idx = m.start()
                    break
        if start_idx >= 0:
            break
        stripped = re.sub(r"^\d+[\.\)]\s*", "", pat_lower).strip()
        if stripped and stripped != pat_lower and len(stripped) > 5:
            search_from = 0
            while True:
                idx = text_lower.find(stripped, search_from)
                if idx < 0:
                    break
                if not _is_toc_entry(max(0, idx - 20)):
                    start_idx = max(0, idx - 20)
                    break
                search_from = idx + len(stripped)
            if start_idx >= 0:
                break

    if start_idx < 0:
        return ""

    end_idx = len(full_text)
    for ep in end_patterns:
        ep_lower = ep.lower()
        idx = text_lower.find(ep_lower, start_idx + 50)
        if idx >= 0 and idx < end_idx:
            end_idx = idx
            continue
        stripped = re.sub(r"^\d+[\.\)]\s*", "", ep_lower).strip()
        if stripped and stripped != ep_lower and len(stripped) > 5:
            idx = text_lower.find(stripped, start_idx + 50)
            if idx >= 0 and idx < end_idx:
                end_idx = idx

    return full_text[start_idx : min(start_idx + max_len, end_idx)]


# Mappa caratteri PDF codificati male (cp850/cp1252 letti come latin-1)
_PDF_CHAR_MAP = {
    "\u00fb": "–",   # û → en-dash
    "\u00da": "è",   # Ú → è
    "\u00c7": "€",   # Ç → €
    "\u00c6": "'",   # Æ → apostrofo
    "\u00de": "è",   # Þ → è
    "\u00c0": "À",   # └ → À
    "\u00d3": "à",   # Ó → à
    "\u2514": "À",   # └ box-drawing → À
}


def _fix_pdf_encoding(s: str) -> str:
    """Corregge artefatti di codifica comuni nei PDF italiani."""
    for old, new in _PDF_CHAR_MAP.items():
        s = s.replace(old, new)
    return s


def _clean(s: str | None) -> str | None:
    """Pulisce spazi multipli, newline e artefatti di codifica PDF."""
    if not s:
        return None
    s = _fix_pdf_encoding(s)
    s = re.sub(r'\*{2,}', '', s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None
