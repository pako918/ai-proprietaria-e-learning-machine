"""
Sezioni H / H-bis – Formato offerta tecnica e contenuto busta / note.

Estrae pagine massime, formato pagina, carattere, interlinea,
contenuto busta tecnica e note importanti sull'offerta tecnica.
"""

from __future__ import annotations
import re
from .utils import _clean, _section_text


def extract_offerta(text: str, text_lower: str) -> dict:
    """
    Restituisce il dict ``offerta_tecnica_formato`` con sotto-chiavi:

    * pagine_massime, formato_pagina, carattere, interlinea
    * esclusi_conteggio
    * contenuto_busta_tecnica  (list[str])
    * note_importanti          (list[str])
    """

    otf: dict = {}

    # ── H) FORMATO OFFERTA TECNICA ───────────────────────────────────────
    m_pag = re.search(
        r"(?:max|massimo|fino\s+a|superare\s+(?:complessivamente\s+)?n\.)\s*(\d+)\s*(?:\([^)]*\)\s*)?pagin\w*",
        text, re.IGNORECASE,
    )
    if m_pag:
        otf["pagine_massime"] = int(m_pag.group(1))

    m_fmt = re.search(r"formato\s+(A[34])", text, re.IGNORECASE)
    if m_fmt:
        otf["formato_pagina"] = m_fmt.group(1).upper()

    m_fac = re.search(r"(\d+)\s*(?:\([^)]*\)\s*)?(?:pagine\s*)?(?:\(?\s*facciate\s*\)?)", text, re.IGNORECASE)
    if m_fac and not otf.get("pagine_massime"):
        otf["pagine_massime"] = int(m_fac.group(1))

    m_escl = re.search(
        r"(?:non\s+rientrano|esclusi|eccetto)[^.]{0,30}conteggio[^.]{0,200}",
        text, re.IGNORECASE,
    )
    if m_escl:
        otf["esclusi_conteggio"] = _clean(m_escl.group(0))[:200]

    m_char = re.search(r"(?:Times\s+New\s+Roman|Arial|Calibri|Garamond)\s*(\d+)", text, re.IGNORECASE)
    if m_char:
        otf["carattere"] = _clean(m_char.group(0))

    m_inter = re.search(r"interlinea\s*[:\s]*([\d]+[.,][\d]+)", text, re.IGNORECASE)
    if m_inter:
        otf["interlinea"] = m_inter.group(1)

    # ── H-bis) CONTENUTO BUSTA TECNICA E NOTE IMPORTANTI ─────────────────

    # --- Contenuto busta tecnica ---
    busta_contenuto: list[str] = []
    busta_section = _section_text(
        text,
        ["BUSTA TECNICA", "CONTENUTO DELLA BUSTA TECNICA",
         r"20\.\s+OFFERTA TECNICA", r"20\.\s+CONTENUTO"],
        ["20.1 RELAZIONE", "21.", "BUSTA ECONOMICA"],
        max_len=5000,
    )
    if busta_section:
        busta_items = re.findall(
            r"(?:^|\n)\s*([a-z])\)\s*(.+?)(?=\n\s*[a-z]\)\s|\n\s*(?:N\.B\.|L.\s*Offerta|La\s+|Il\s+|Si\s+)|$)",
            busta_section, re.DOTALL,
        )
        for _, item_text in busta_items:
            cleaned = _clean(item_text.replace("\n", " "))[:300]
            if cleaned and len(cleaned) > 5:
                busta_contenuto.append(cleaned)

    if not busta_contenuto:
        m_busta = re.search(
            r"[Bb]usta\s+[Tt]ecnica.*?(?:deve\s+contenere|conterr.)[^:]*:\s*(.*?)(?=\n\s*(?:N\.B\.|L.\s*Offerta|La\s+Relazione|20\.\d))",
            text, re.DOTALL,
        )
        if m_busta:
            items_block = m_busta.group(1)
            for m_item in re.finditer(r"([a-z])\)\s*(.+?)(?=\s*[a-z]\)\s|$)", items_block, re.DOTALL):
                cleaned = _clean(m_item.group(2).replace("\n", " "))[:300]
                if cleaned and len(cleaned) > 5:
                    busta_contenuto.append(cleaned)
    otf["contenuto_busta_tecnica"] = busta_contenuto

    # --- Note importanti offerta tecnica ---
    note_ot: list[str] = []
    ot_section = _section_text(
        text,
        [r"20\.\s+OFFERTA TECNICA", r"20\.\s+CONTENUTO DELL", "BUSTA TECNICA"],
        [r"21\.", "BUSTA ECONOMICA", r"21\.\s+OFFERTA ECONOMICA"],
        max_len=10000,
    )
    search_text = ot_section if ot_section.strip() else text
    search_text_clean = re.sub(r'\*{2,}', ' ', search_text)
    search_text_clean = re.sub(r'\s+', ' ', search_text_clean)

    # Offerta identica per entrambi i lotti
    m_identica = re.search(
        r"(?:offerta\s+tecnica\s+(?:deve\s+essere\s+)?identica\s+per\s+entrambi\s+i\s+lotti|"
        r"identica\s+per\s+entrambi\s+i\s+lotti)[^.]*\.\s*(?:In\s+caso\s+di\s+difformit(?:[^.]|\.\s*(?=\d))*\.)?",
        search_text_clean, re.IGNORECASE | re.DOTALL,
    )
    if m_identica:
        note_ot.append(_clean(m_identica.group(0).replace("\n", " "))[:400])

    # Caratteristiche minime / principio di equivalenza
    m_equiv = re.search(
        r"L.Offerta\s+Tecnica\s+deve\s+rispettare\s+le\s+caratteristiche\s+minime[^.]*\.",
        search_text_clean, re.DOTALL,
    )
    if m_equiv:
        note_ot.append(_clean(m_equiv.group(0).replace("\n", " "))[:400])

    # No indicazioni economiche (a pena di esclusione)
    m_noeco = re.search(
        r"L.Offerta\s+Tecnica[^.]*?a\s+pena\s+di\s+esclusione[^.]*?economic\w+[^.]*\.",
        search_text_clean, re.IGNORECASE | re.DOTALL,
    )
    if m_noeco:
        note_ot.append(_clean(m_noeco.group(0).replace("\n", " "))[:400])

    # Commissione giudicatrice chiarimenti
    m_comm = re.search(
        r"[Ll]a\s+Commissione\s+giudicatrice[^.]*?chiarimenti[^.]*\.",
        search_text_clean, re.IGNORECASE | re.DOTALL,
    )
    if m_comm:
        note_ot.append(_clean(m_comm.group(0).replace("\n", " "))[:400])
    otf["note_importanti"] = note_ot

    return otf
