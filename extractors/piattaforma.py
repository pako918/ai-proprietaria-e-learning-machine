"""Sezione C: Piattaforma telematica."""

import re


def extract_piattaforma(text: str, text_lower: str) -> dict:
    """Estrae nome/gestore/URL della piattaforma telematica."""
    pt = {}

    piattaforme_map = {
        "sintel": ("Sintel", "ARIA S.p.A."),
        "tuttogare": ("TuttoGare", None),
        "net4market": ("Net4market", None),
        "mepa": ("MePA", "Consip"),
        "acquistinretepa": ("AcquistinretePa", "Consip"),
        "start": ("START", "Regione Toscana"),
        "s.tel.la": ("S.TEL.LA.", "Regione Lazio"),
        "stella": ("S.TEL.LA.", "Regione Lazio"),
        "traspare": ("TrasparE", None),
        "sardegna cat": ("SardegnaCat", "Regione Sardegna"),
        "empulia": ("EmPULIA", "InnovaPuglia"),
        "gare telematiche": ("Gare Telematiche", None),
        "appalti&contratti": ("Appalti&Contratti", None),
        "portale appalti": ("Portale Appalti", None),
        "maggioli": ("Maggioli", None),
        "asmecomm": ("Asmecomm", None),
        "intercent-er": ("Intercent-ER", "Regione Emilia-Romagna"),
        "intercenter": ("Intercent-ER", "Regione Emilia-Romagna"),
        "ariaspa": ("Sintel", "ARIA S.p.A."),
        "albofornitori": ("Albo Fornitori", None),
        "e-procurement": ("e-Procurement", None),
    }
    for key, (nome, gestore) in piattaforme_map.items():
        if re.search(r'\b' + re.escape(key) + r'\b', text_lower):
            pt["nome"] = nome
            if gestore:
                pt["gestore"] = gestore
            break

    # Match URL by platform name in the HOSTNAME only (prevent false positives
    # from keywords appearing in URL paths, e.g. "mepa" inside "HomePage.jsp").
    m_url = re.search(
        r"https?://[^/\s\n]*(?:aria|sintel|tuttogare|net4market|traspare|acquistinretepa|start\.toscana|empulia|intercent|maggioli|asmecomm|portaleappalti|mepa)[^/\s\n]*(?:/[^\s\n]*)?",
        text, re.IGNORECASE)
    if m_url:
        pt["url"] = m_url.group(0).rstrip(".,;)")
    elif not pt.get("url"):
        m_url_gen = re.search(r"(?:piattaforma|telematica)[^.]{0,200}?(https?://[^\s\n]+)", text, re.IGNORECASE)
        if m_url_gen:
            pt["url"] = m_url_gen.group(1).rstrip(".,;)")

    return pt
