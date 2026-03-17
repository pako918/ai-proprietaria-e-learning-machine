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

    # Look near "procedura negoziata/aperta" first — avoids picking up
    # unrelated D.Lgs. references (e.g. art. 98 D.Lgs. 81/2008 for sicurezza).
    m_ref = None
    m_proc_ctx = re.search(
        r"procedura\s+negoziata[^.]{0,400}?"
        r"(?:ai\s+sensi\s+(?:dell[\u2018\u2019']\s*)?|dell[\u2018\u2019']\s*|ex\s+)"
        r"(art\.?\s*\d+[^;\n]{0,200}?D\.?[Ll]gs\.?\s*(?:n\.?\s*)?\d+/\d{4})",
        text, re.IGNORECASE | re.DOTALL,
    )
    if m_proc_ctx:
        tp["riferimento_normativo"] = _clean(m_proc_ctx.group(1))
    else:
        # Prefer D.Lgs. 36/2023 (codice appalti moderno) for negoziata sotto soglia
        m_art50 = re.search(
            r"(art\.?\s*50[^;\n]{0,200}?D\.?[Ll]gs\.?\s*(?:n\.?\s*)?36/2023)",
            text, re.IGNORECASE,
        )
        if m_art50:
            tp["riferimento_normativo"] = _clean(m_art50.group(1))
        else:
            m_ref36 = re.search(
                r"(?:ai sensi|ex)\s+(art\.?\s*\d+[^.]{0,80}D\.?[Ll]gs\.?\s*(?:n\.?\s*)?36/2023)",
                text, re.IGNORECASE,
            )
            if m_ref36:
                tp["riferimento_normativo"] = _clean(m_ref36.group(1))
            else:
                m_ref_gen = re.search(
                    r"(?:ai sensi|ex)\s+(art\.?\s*\d+[^.]{0,80}D\.?Lgs\.?\s*(?:n\.?\s*)?\d+/\d{4})",
                    text, re.IGNORECASE,
                )
                if m_ref_gen:
                    rif = _clean(m_ref_gen.group(1))
                    # Skip security-coordinator reference (D.Lgs. 81/2008 art.89/98/100)
                    if "81/2008" not in rif:
                        tp["riferimento_normativo"] = rif

    if tp.get("tipo") == "negoziata" and "senza bando" in text_lower:
        tp["senza_bando"] = True

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
