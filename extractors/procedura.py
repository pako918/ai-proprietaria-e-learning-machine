"""Sezione B: Tipo procedura."""

import re
from .utils import _clean


def extract_procedura(text: str, text_lower: str) -> dict:
    """Estrae tipo procedura, criterio aggiudicazione, accordo quadro, ecc."""
    tp = {}

    if "procedura aperta" in text_lower:
        tp["tipo"] = "aperta"
    elif "procedura negoziata" in text_lower:
        tp["tipo"] = "negoziata"
    elif "procedura ristretta" in text_lower:
        tp["tipo"] = "ristretta"
    elif "procedura competitiva" in text_lower:
        tp["tipo"] = "competitiva_con_negoziazione"

    if "sotto soglia" in text_lower or "sotto la soglia" in text_lower or "sottosoglia" in text_lower:
        tp["ambito"] = "sotto_soglia"
    elif "sopra soglia" in text_lower or "comunitaria" in text_lower:
        tp["ambito"] = "sopra_soglia_comunitaria"

    if "minor prezzo" in text_lower or "prezzo più basso" in text_lower:
        tp["criterio_aggiudicazione"] = "minor_prezzo"
    elif "oepv" in text_lower or "offerta economicamente più vantaggiosa" in text_lower or "miglior rapporto qualità" in text_lower:
        tp["criterio_aggiudicazione"] = "OEPV"

    if tp.get("criterio_aggiudicazione") == "OEPV":
        if "miglior rapporto qualità" in text_lower or "qualità/prezzo" in text_lower:
            tp["metodo_OEPV"] = "miglior_rapporto_qualita_prezzo"

    m_ref = re.search(r"(?:ai sensi|ex)\s+(art\.?\s*\d+[^.]{0,80}D\.?Lgs\.?\s*(?:n\.?\s*)?\d+/\d{4})", text, re.IGNORECASE)
    if m_ref:
        tp["riferimento_normativo"] = _clean(m_ref.group(1))

    if "inversione procedimentale" in text_lower or "art. 107 comma 3" in text_lower or "articolo 107, comma 3" in text_lower:
        tp["inversione_procedimentale"] = True

    if "accordo quadro" in text_lower or "accordi quadro" in text_lower:
        aq = {"presente": True}
        if "unico operatore" in text_lower:
            aq["tipo"] = "unico_operatore"
        elif "più operatori" in text_lower:
            aq["tipo"] = "piu_operatori"
        m_dur = re.search(r"(?:accordo quadro|durata)[^.]{0,100}?(\d+)\s*mesi", text_lower)
        if m_dur:
            aq["durata_mesi"] = int(m_dur.group(1))
        tp["accordo_quadro"] = aq
    else:
        tp["accordo_quadro"] = {"presente": False}

    if "concessione" in text_lower and ("art. 182" in text_lower or "art. 194" in text_lower):
        tp["concessione"] = True

    return tp
