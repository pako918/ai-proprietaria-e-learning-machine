"""Sezione E: Tempistiche."""

import re
from .utils import _clean, _parse_euro


def extract_tempistiche(text: str, text_lower: str) -> tuple[dict, dict | None]:
    """Estrae scadenze, durate, fasi.

    Returns:
        (temp, durata_contratto) — tempistiche e eventuale durata_contratto.
    """
    temp = {}
    durata_contratto = None

    # Scadenza offerte
    scad_patterns = [
        r"(?:entro\s+(?:e\s+non\s+oltre\s+)?(?:le\s+)?ore\s+)(\d{1,2}[:.,:]\d{2})\s+del\s+(?:giorno\s+)?(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})",
        r"(?:entro\s+il\s+(?:giorno\s+)?)(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})\s+(?:ore\s+)?(\d{1,2}[:.]\d{2})",
        r"(?:scadenza|termine)[^.]{0,200}?(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})\s+(?:ore\s+)?(\d{1,2}[:.]\d{2})",
        r"presentazione\s+(?:dell['\u2019]\s*)?offert\w+[^.]{0,150}?(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})\s+(?:ore\s+)?(\d{1,2}[:.]\d{2})",
        r"(?:scadenza|termine)\s+(?:per\s+la\s+)?(?:presentazione|ricezione)\s+(?:dell['\u2019]\s*)?offert\w+[^.]{0,100}?(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})",
        r"offert\w+\s+(?:dovranno|devono)\s+(?:essere\s+)?(?:presentat\w+|trasmess\w+|inviat\w+|pervenire)[^.]{0,200}?(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})(?:\s+(?:ore\s+)?(\d{1,2}[:.]\d{2}))?",
        r"(?:entro\s+(?:e\s+non\s+oltre\s+)?(?:le\s+)?ore\s+)(\d{1,2}[:.,:]\d{2})\s+del\s+(\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4})",
        r"(?:scadenza|termine)[^.]{0,60}?(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})",
    ]
    for sp in scad_patterns:
        m_scad = re.search(sp, text, re.IGNORECASE | re.DOTALL)
        if m_scad:
            groups = m_scad.groups()
            if len(groups) >= 2 and groups[1]:
                g1, g2 = groups[0], groups[1]
                if re.match(r"\d{1,2}[:.,:]\d{2}$", g1) and re.match(r"\d{1,2}[/.\-]", g2):
                    temp["scadenza_offerte"] = f"{g2} ore {g1}"
                else:
                    temp["scadenza_offerte"] = f"{g1} ore {g2}"
            else:
                temp["scadenza_offerte"] = groups[0]
            break

    # Termine chiarimenti
    chiar_patterns = [
        r"(?:chiariment\w+|quesit\w+)[^.]{0,200}?entro\s+(?:e\s+non\s+oltre\s+)?(?:il\s+)?(?:giorno\s+)?(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})",
        r"(?:termine|scadenza)\s+(?:per\s+)?(?:la\s+)?(?:presentazione\s+(?:dei\s+)?)?(?:chiariment\w+|quesit\w+)[^.]{0,100}?(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})",
        r"(?:richiesta\s+di\s+)?chiariment\w+[^.]{0,150}?(?:entro\s+il\s+)(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})",
    ]
    for cp in chiar_patterns:
        m_chiar = re.search(cp, text, re.IGNORECASE | re.DOTALL)
        if m_chiar:
            temp["termine_chiarimenti"] = m_chiar.group(1)
            break

    # Apertura buste
    m_apertura = re.search(
        r"(?:prima\s+sedut|apertura\s+(?:delle\s+)?buste|seduta\s+pubblica)[^.]{0,200}?(?:il\s+(?:giorno\s+)?)?(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})(?:\s+(?:ore\s+|alle\s+)?(\d{1,2}[:.]\d{2}))?",
        text, re.IGNORECASE | re.DOTALL,
    )
    if m_apertura:
        apertura = m_apertura.group(1)
        if m_apertura.group(2):
            apertura += f" ore {m_apertura.group(2)}"
        temp["apertura_buste"] = apertura

    # Validità offerta
    m_val = re.search(r"validit[àa]\s+(?:dell['\u2019]\s*)?offert\w+[^.]{0,50}?(\d+)\s*(?:giorni|gg)", text, re.IGNORECASE)
    if m_val:
        temp["validita_offerta_giorni"] = int(m_val.group(1))
    elif "180 giorni" in text_lower:
        temp["validita_offerta_giorni"] = 180

    # Durata procedimento
    m_dur_proc = re.search(r"durata\s+(?:massima\s+)?del(?:la)?\s+procedur\w+[^.]{0,80}?(\d+)\s*mesi", text, re.IGNORECASE)
    if m_dur_proc:
        temp["durata_procedimento_mesi"] = int(m_dur_proc.group(1))

    # Durata contratto / concessione
    dur_c = {}
    m_dur_anni = re.search(
        r"(?:durat\w+\s+(?:del(?:la)?\s+)?(?:contratto|concessione|appalto|servizio|incarico)|"
        r"(?:gestione|opera)\s+[eè]\s+concess\w+\s+per)"
        r"[^.]{0,120}?(\d+)\s*ann[io]",
        text, re.IGNORECASE,
    )
    if m_dur_anni:
        dur_c["anni"] = int(m_dur_anni.group(1))

    m_dur_gen = re.search(
        r"durat\w+\s+(?:complessiv\w+\s+)?(?:dell?['\u2019]\s*)?(?:contratto|appalto|servizio|incarico|lotto|procedura|prestazion\w+)"
        r"[^.]{0,150}?(\d+)\s*(?:giorni|gg)\s+(?:naturali\s+e\s+consecutivi|natural\w+|lavora\w+|solari\w+)?",
        text, re.IGNORECASE,
    )
    if not m_dur_gen:
        m_dur_gen = re.search(
            r"durat\w+\s+complessiv\w+[^.]{0,200}?(?:stimat\w+\s+in|pari\s+a|di)\s*(\d+)\s*(?:giorni|gg)",
            text, re.IGNORECASE,
        )
    if m_dur_gen:
        dur_c["giorni"] = int(m_dur_gen.group(1))
        temp["durata_esecuzione_giorni"] = dur_c["giorni"]

    m_dur_m = re.search(r"durat\w+\s+(?:dell?['\u2019]\s*)?(?:contratto|appalto|servizio)[^.]{0,100}?(\d+)\s*mesi", text, re.IGNORECASE)
    if m_dur_m:
        dur_c["mesi"] = int(m_dur_m.group(1))
        temp["durata_esecuzione_mesi"] = dur_c["mesi"]

    m_fase_lav = re.search(r"(?:fase\s+(?:II|2|di\s+)?(?:realizzazione|lavori|esecuz))\w*[^.]{0,100}?(\d+)\s*(?:mesi|anni)", text, re.IGNORECASE)
    if m_fase_lav:
        unit = "anni" if "ann" in m_fase_lav.group(0).lower() else "mesi"
        dur_c[f"fase_realizzazione_{unit}"] = int(m_fase_lav.group(1))

    m_fase_gest = re.search(r"(?:fase\s+(?:III|3|di\s+)?(?:gestion|conduzion|manuten))\w*[^.]{0,100}?(\d+)\s*(?:mesi|anni)", text, re.IGNORECASE)
    if not m_fase_gest:
        m_fase_gest = re.search(r"(?:gestion\w+\s+dell['\u2019]?\s*[Oo]pera)\s+[^\d]{0,30}?(?:concess\w+\s+)?(?:per\s+)?(\d+)\s*ann", text, re.IGNORECASE)
    if m_fase_gest:
        unit = "anni" if "ann" in m_fase_gest.group(0).lower() else "mesi"
        dur_c[f"fase_gestione_{unit}"] = int(m_fase_gest.group(1))

    if dur_c:
        durata_contratto = dur_c

    # Società di scopo
    if "società di scopo" in text_lower:
        sds = {"obbligatoria": True}
        m_sds_forma = re.search(
            r"societ[àa]\s+di\s+scopo\s+(?:in\s+)?(?:forma\s+di\s+)?(societ[àa]\s+per\s+azioni\s+o\s+a\s+responsabilit[àa]\s+limitata[^.]{0,60})",
            text, re.IGNORECASE,
        )
        if m_sds_forma:
            sds["forma"] = _clean(m_sds_forma.group(1))
        m_sds_cap = re.search(r"capitale\s+sociale\s+minimo[^.]{0,60}?([\d.,]+)\s*(?:€|euro)", text, re.IGNORECASE)
        if m_sds_cap:
            v = _parse_euro(m_sds_cap.group(1))
            if v:
                sds["capitale_sociale_minimo"] = v
        if durata_contratto is None:
            durata_contratto = {}
        durata_contratto["societa_di_scopo"] = sds

    # Termine stipula
    m_stip = re.search(r"stipula[^.]{0,100}?(\d+)\s*giorni", text, re.IGNORECASE)
    if m_stip:
        temp["termine_stipula_giorni"] = int(m_stip.group(1))

    # Standstill
    m_stand = re.search(r"stand\s*[-]?\s*still[^.]{0,80}?(\d+)\s*giorni", text, re.IGNORECASE)
    if m_stand:
        temp["standstill_giorni"] = int(m_stand.group(1))

    return temp, durata_contratto
