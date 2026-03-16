"""
Sezioni J-U – Sezioni complementari.

Raggruppa le sezioni più piccole:

* J) Subappalto
* K) Avvalimento
* L) Sopralluogo
* M) Documentazione amministrativa
* N) Cause di esclusione
* O) Aggiudicazione e stipula
* P) Penali
* Q) Sicurezza
* R) CAM
* S) Controversie
* T) Commissione giudicatrice
* U) CCNL, Fonti finanziamento, Condizioni PNRR
"""

from __future__ import annotations
import re
from .utils import _clean, _parse_euro, _section_text


def extract_complementari(text: str, text_lower: str) -> dict:
    """
    Restituisce un dict con le chiavi di primo livello:

    subappalto, avvalimento, sopralluogo, documentazione_amministrativa,
    soccorso_istruttorio, cause_esclusione, aggiudicazione, penali,
    sicurezza, CAM_criteri_ambientali, controversie, informazioni_aggiuntive.

    Il chiamante deve fare ``result.update(...)`` con il risultato.
    """

    out: dict = {}

    # ══════════════════════════════════════════════════════════════════════
    # J) SUBAPPALTO
    # ══════════════════════════════════════════════════════════════════════
    sub: dict = {}
    sub_section = _section_text(
        text,
        ["SUBAPPALTO"],
        ["9.", "REQUISITI DI PARTECIPAZIONE", "GARANZI", "SOPRALLUOGO"],
        max_len=5000,
    )
    if sub_section:
        sub["ammesso"] = True
        m_sub_perc = re.search(
            r"(\d+)\s*%\s*(?:dell?\s*['\u2019]?\s*import|del\s+valore|del\s+contratto)",
            sub_section, re.IGNORECASE,
        )
        if m_sub_perc:
            sub["limite_percentuale"] = int(m_sub_perc.group(1))
        if "non è ammesso" in sub_section.lower() or "non ammesso" in sub_section.lower():
            sub["ammesso"] = False
    elif "subappalto" in text_lower:
        sub["ammesso"] = True
    out["subappalto"] = sub

    # ══════════════════════════════════════════════════════════════════════
    # K) AVVALIMENTO
    # ══════════════════════════════════════════════════════════════════════
    avv: dict = {}
    avv_section = _section_text(
        text,
        ["AVVALIMENTO"],
        ["8. SUBAPPALTO", "SUBAPPALTO", "REQUISITI", "GARANZI"],
        max_len=3000,
    )
    if avv_section:
        avv["ammesso"] = True
        if "non è ammesso" in avv_section.lower() or "non ammesso" in avv_section.lower():
            avv["ammesso"] = False
    out["avvalimento"] = avv

    # ══════════════════════════════════════════════════════════════════════
    # L) SOPRALLUOGO
    # ══════════════════════════════════════════════════════════════════════
    sop: dict = {}
    sop_section = _section_text(
        text,
        ["SOPRALLUOGO"],
        ["12.", "PAGAMENTO", "CONTRIBUTO", "MODALITÀ", "13."],
        max_len=3000,
    )
    if sop_section:
        sop_lower = sop_section.lower()
        if "obbligatorio" in sop_lower:
            if "non obbligatorio" in sop_lower or "non è obbligatorio" in sop_lower or "non previsto" in sop_lower:
                sop["obbligatorio"] = False
            else:
                sop["obbligatorio"] = True
        elif "non previsto" in sop_lower or "non richiesto" in sop_lower:
            sop["obbligatorio"] = False
        elif "facoltativo" in sop_lower:
            sop["obbligatorio"] = False
            sop["note"] = "Facoltativo"

        m_sop_mod = re.search(
            r"(?:sopralluogo)[^.]{0,200}?(?:da\s+effettuar\w+|si\s+svol\w+|mediante|con\s+modalit\w+)\s+([^.]{10,200})",
            sop_section, re.IGNORECASE | re.DOTALL,
        )
        if m_sop_mod:
            sop["modalita"] = _clean(m_sop_mod.group(1))[:200]

        m_sop_cont = re.search(
            r"(?:prenotazion\w+|appuntamento|contattare|richied\w+)[^.]{0,100}?([\w.]+@[\w.]+\.\w{2,}|\d{2,4}[/-]?\d{4,})",
            sop_section, re.IGNORECASE,
        )
        if m_sop_cont:
            sop["contatti_prenotazione"] = _clean(m_sop_cont.group(1))

        m_sop_term = re.search(
            r"sopralluogo[^.]{0,200}?entro\s+(?:il\s+)?(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})",
            sop_section, re.IGNORECASE | re.DOTALL,
        )
        if m_sop_term:
            sop["termine"] = m_sop_term.group(1)
    # Full-text fallback when no section was found
    if not sop:
        _sop_neg = re.search(
            r"sopralluogo[^.\n]{0,200}?(?:non\s+(?:è\s+)?(?:previsto|richiesto|obbligatorio)|"
            r"\bnon\s+previsto\b|\bnessun\s+sopralluogo\b)",
            text, re.IGNORECASE,
        )
        if _sop_neg:
            sop["obbligatorio"] = False
    out["sopralluogo"] = sop

    # ══════════════════════════════════════════════════════════════════════
    # M) DOCUMENTAZIONE AMMINISTRATIVA
    # ══════════════════════════════════════════════════════════════════════
    doc: dict = {"contributo_ANAC": {}, "imposta_bollo": {}}

    # Contributo ANAC
    anac = doc["contributo_ANAC"]
    anac_patterns = [
        r"contributo\s+(?:a\s+favore\s+dell?\s*['\u2019]?\s*)?ANAC[^€\d]{0,100}?[€\s]*([\d.,]+)",
        r"contributo\s+(?:previsto\s+)?(?:dalla\s+legge\s+)?(?:in\s+favore\s+)?(?:dell?\s*['\u2019]?\s*)?(?:Autorit[àa]\s+Nazionale\s*\n?\s*Anticorruzione|ANAC)[^€\d]{0,200}?(?:€|euro)\s*\.?\s*([\d.,]+)",
        r"(?:€|euro)\s*\.?\s*([\d.,]+)[^\n]{0,50}?(?:contributo|ANAC)",
        r"contributo\s+ANAC[^\d]{0,60}?([\d.,]+)",
    ]
    for anac_pat in anac_patterns:
        m_anac = re.search(anac_pat, text, re.IGNORECASE | re.DOTALL)
        if m_anac:
            v = _parse_euro(m_anac.group(1))
            if v and v >= 20:
                anac["dovuto"] = True
                anac["importo_totale"] = v
                break
    if "importo_totale" not in anac and "contributo" in text_lower and "anac" in text_lower:
        anac["dovuto"] = True

    anac_lotti = re.findall(r"lotto\s+\d[^€]{0,30}?[€\s]*([\d.,]+)\s*(?:€|\n)", text, re.IGNORECASE)
    if anac_lotti and len(anac_lotti) >= 2:
        anac["importi_per_lotto"] = [
            {"lotto": i + 1, "importo": _parse_euro(v)}
            for i, v in enumerate(anac_lotti)
            if _parse_euro(v)
        ]

    # Imposta di bollo
    bollo = doc["imposta_bollo"]
    m_bollo = re.search(
        r"(?:imposta\s+di\s+bollo|bollo)[^.]{0,100}?(?:€|Ç|euro|pari\s+a)\s*([\d.,]+)",
        text, re.IGNORECASE,
    )
    if m_bollo:
        v = _parse_euro(m_bollo.group(1))
        if v:
            bollo["importo"] = v

    # DGUE
    if "dgue" in text_lower or "espd" in text_lower:
        doc["DGUE"] = {"richiesto": True}
        if "firma digitale" in text_lower:
            doc["DGUE"]["firma"] = "digitale"

    # Documentazione richiesta — conta documenti
    doc_section = _section_text(text, [
        "DOCUMENTAZIONE AMMINISTRATIVA", "BUSTA A",
        "14. CONTENUTO", "19. CONTENUTO", "CONTENUTO DELLA BUSTA",
    ], [
        "OFFERTA TECNICA", "BUSTA B", "BUSTA TECNICA",
        r"20\.", r"15\.",
    ], max_len=15000)
    if doc_section:
        doc_items = re.findall(r"(?:^|\n)\s*(\d{1,2})\s*[\.\)\-]\s+([^\n]{10,200})", doc_section)
        valid_items = [(int(n), desc) for n, desc in doc_items if 1 <= int(n) <= 15]
        if valid_items:
            max_n = max(n for n, _ in valid_items)
            if max_n <= 15:
                doc["numero_documenti_richiesti"] = max_n

    out["documentazione_amministrativa"] = doc

    # ── Soccorso istruttorio ─────────────────────────────────────────────
    si: dict = {}
    if "soccorso istruttorio" in text_lower:
        si["ammesso"] = True
        si_patterns = [
            r"soccorso\s+istruttorio.{0,300}?termine\s+di\s+(\d+)\s*(?:\([^)]+\)\s*)?giorni",
            r"soccorso\s+istruttorio.{0,300}?(\d+)\s*(?:\([^)]+\)\s*)?giorni",
            r"termine.{0,50}?soccorso\s+istruttorio.{0,50}?(\d+)\s*giorni",
        ]
        for si_pat in si_patterns:
            m_si = re.search(si_pat, text, re.IGNORECASE | re.DOTALL)
            if m_si:
                val_si = int(m_si.group(1))
                if 1 <= val_si <= 30:
                    si["termine_giorni"] = val_si
                    break
        m_si_ref = re.search(r"art\.?\s*(101|83)[^.]{0,30}D\.?Lgs", text, re.IGNORECASE)
        if m_si_ref:
            si["riferimento"] = _clean(m_si_ref.group(0))
    out["soccorso_istruttorio"] = si

    # ══════════════════════════════════════════════════════════════════════
    # N) CAUSE DI ESCLUSIONE
    # ══════════════════════════════════════════════════════════════════════
    ce: dict = {}
    if "art. 94" in text or "art.94" in text or "articolo 94" in text_lower:
        ce["automatiche"] = {"riferimento": "art. 94 D.Lgs. 36/2023"}
    if "art. 95" in text or "art.95" in text or "articolo 95" in text_lower:
        ce["non_automatiche"] = {"riferimento": "art. 95 D.Lgs. 36/2023"}
    if "self cleaning" in text_lower or "self-cleaning" in text_lower or "art. 96" in text:
        ce["self_cleaning"] = {"ammesso": True, "riferimento": "art. 96 D.Lgs. 36/2023"}
    out["cause_esclusione"] = ce

    # ══════════════════════════════════════════════════════════════════════
    # O) AGGIUDICAZIONE E STIPULA
    # ══════════════════════════════════════════════════════════════════════
    agg: dict = {"stipula_contratto": {}}
    m_max_lotti = re.search(r"(?:massimo|max)\s+(\d+)\s+lott[oi]", text, re.IGNORECASE)
    if m_max_lotti:
        agg["numero_lotti_massimi_per_concorrente"] = int(m_max_lotti.group(1))
    elif "un solo lotto" in text_lower or "aggiudicatario di un solo lotto" in text_lower:
        agg["numero_lotti_massimi_per_concorrente"] = 1

    stip = agg["stipula_contratto"]
    m_stip_forma = re.search(
        r"contratto\s+(?:è|Þ|sara)\s+stipulat\w+\s+(?:mediante|in\s+forma\s+di|per)\s+([^.]{5,80})\.?",
        text, re.IGNORECASE,
    )
    if m_stip_forma:
        forma_raw = _clean(m_stip_forma.group(1)) or ""
        if "scrittura privata" in forma_raw.lower():
            stip["forma"] = "scrittura_privata"
            stip["dettaglio"] = forma_raw
        elif "atto pubblico" in forma_raw.lower():
            stip["forma"] = "atto_pubblico"
        elif "forma pubblica" in forma_raw.lower():
            stip["forma"] = "forma_pubblica_amministrativa"
        else:
            stip["forma"] = "altro"
            stip["dettaglio"] = forma_raw
    elif "forma pubblica amministrativa" in text_lower:
        stip["forma"] = "forma_pubblica_amministrativa"
    elif "scrittura privata" in text_lower:
        stip["forma"] = "scrittura_privata"
    elif "atto pubblico" in text_lower:
        stip["forma"] = "atto_pubblico"

    m_stand = re.search(r"stand\s*-?\s*still[^.]{0,50}?(\d+)\s*giorni", text, re.IGNORECASE)
    if not m_stand:
        m_stand = re.search(r"(\d+)\s*giorni[^.]{0,30}comunicazione[^.]{0,30}aggiudicazione", text, re.IGNORECASE)
    if m_stand:
        stip["termine_standstill_giorni"] = int(m_stand.group(1))
    out["aggiudicazione"] = agg

    # ══════════════════════════════════════════════════════════════════════
    # P) PENALI
    # ══════════════════════════════════════════════════════════════════════
    pen: dict = {}
    m_pen = re.search(
        r"penal[ei][^.]{0,200}?(\d+(?:[.,]\d+)?)\s*%\s*(?:(?:al\s+)?giorn|per\s+ogni\s+giorn)",
        text, re.IGNORECASE,
    )
    if m_pen:
        pen["previste"] = True
        pen["percentuale_giornaliera"] = float(m_pen.group(1).replace(",", "."))
    m_pen_max = re.search(
        r"penal[ei][^.]{0,300}?(?:massim\w+|tetto|limit\w+)[^.]{0,50}?(\d+(?:[.,]\d+)?)\s*%",
        text, re.IGNORECASE,
    )
    if m_pen_max:
        pen["tetto_massimo_percentuale"] = float(m_pen_max.group(1).replace(",", "."))
    out["penali"] = pen

    # ══════════════════════════════════════════════════════════════════════
    # Q) SICUREZZA
    # ══════════════════════════════════════════════════════════════════════
    sic: dict = {}
    info_agg: dict = {"note": []}  # informazioni_aggiuntive (populated also in T/U)

    if "natura intellettuale" in text_lower:
        sic["DUVRI"] = {"richiesto": False, "nota": "Servizio di natura intellettuale, nessun rischio interferenza"}
        info_agg["natura_servizio"] = "natura intellettuale"
    elif re.search(
        r"non\s+(?:sussiste|è\s+richiest\w+|è\s+necessari\w+|previsto)\b[^.]{0,60}duvri|"
        r"duvri[^.]{0,60}non\s+(?:richiest\w+|necessari\w+|previst\w+|dovut\w+)",
        text, re.IGNORECASE,
    ):
        sic["DUVRI"] = {"richiesto": False, "nota": "DUVRI non richiesto come da disciplinare"}
    elif "duvri" in text_lower:
        sic["DUVRI"] = {"richiesto": True}
    out["sicurezza"] = sic

    # ══════════════════════════════════════════════════════════════════════
    # R) CAM
    # ══════════════════════════════════════════════════════════════════════
    cam: dict = {}
    if "criteri ambientali minimi" in text_lower or "cam" in text_lower:
        cam["applicabili"] = True
        m_cam = re.search(
            r"(?:decreto|D\.?M\.?)\s*(?:n\.?\s*)?(\d+)\s+del\s+(\d{1,2}\s+\w+\s+\d{4})",
            text, re.IGNORECASE,
        )
        if m_cam:
            cam["decreto_riferimento"] = f"D.M. n. {m_cam.group(1)} del {m_cam.group(2)}"
    out["CAM_criteri_ambientali"] = cam

    # ══════════════════════════════════════════════════════════════════════
    # S) CONTROVERSIE
    # ══════════════════════════════════════════════════════════════════════
    cont: dict = {}
    m_tar = re.search(r"TAR\s+(?:di\s+)?(\w+(?:\s+\w+)?)", text, re.IGNORECASE)
    if m_tar:
        cont["foro_competente"] = f"TAR {_clean(m_tar.group(1))}"

    m_foro = re.search(r"[Ff]oro\s+(?:competente\s+)?(?:di\s+|è\s+)(\w+(?:\s+\w+)?)", text)
    if m_foro:
        cont["giurisdizione_contratto"] = f"Foro di {_clean(m_foro.group(1))}"

    if "arbitrato" in text_lower:
        if "escluso" in text_lower[text_lower.find("arbitrato"):text_lower.find("arbitrato") + 100]:
            cont["arbitrato"] = "escluso"
        else:
            cont["arbitrato"] = "ammesso"
    out["controversie"] = cont

    # ══════════════════════════════════════════════════════════════════════
    # T) COMMISSIONE GIUDICATRICE
    # ══════════════════════════════════════════════════════════════════════
    m_comm = re.search(r"commissione[^.]{0,100}?(\d+)\s*(?:membri|componenti)", text, re.IGNORECASE)
    if m_comm:
        info_agg["commissione_giudicatrice"] = {
            "prevista": True,
            "numero_membri": int(m_comm.group(1)),
        }

    if "codice di comportamento" in text_lower:
        info_agg["codice_comportamento"] = True

    if "tracciabilità" in text_lower and ("flussi" in text_lower or "136/2010" in text or "legge 136" in text_lower):
        info_agg["tracciabilita_flussi"] = True

    # ══════════════════════════════════════════════════════════════════════
    # U) CCNL, FONTI FINANZIAMENTO, CONDIZIONI PNRR
    # ══════════════════════════════════════════════════════════════════════
    m_ccnl = re.search(r"(?:CCNL|C\.C\.N\.L\.?|contratto\s+collettivo\s+nazionale)[^\n]{0,200}", text, re.IGNORECASE)
    if m_ccnl:
        ccnl_text = _clean(m_ccnl.group(0))
        if ccnl_text and len(ccnl_text) > 10:
            info_agg["CCNL"] = ccnl_text[:300]

    fonti: list[str] = []
    fonti_patterns = [
        (r"PNRR", "PNRR"),
        (r"M\.?7.*?Repower|Misura\s+7|M7\s*[-.]?\s*1\.7", "PNRR M.7-1.7 Repower"),
        (r"Fondo\s+Complementare|PNC", "Fondo Complementare PNC"),
        (r"Superbonus|110\s*%|Ecobonus", "Superbonus/Ecobonus"),
        (r"Conto\s+Termico", "Conto Termico"),
        (r"certificati?\s+bianch\w+|TEE", "Certificati Bianchi/TEE"),
        (r"GSE|Gestore\s+Servizi\s+Energetici", "GSE"),
        (r"risorse\s+propri\w+\s+(?:del\s+)?(?:concession|operat)", "Risorse proprie del concessionario"),
        (r"fondi?\s+(?:di\s+)?[Bb]ilancio", "Fondi di Bilancio"),
        (r"fondi?\s+(?:region\w+|POR|FSC|FESR)", "Fondi regionali/europei"),
    ]
    for pat, label in fonti_patterns:
        if re.search(pat, text, re.IGNORECASE):
            fonti.append(label)
    if fonti:
        info_agg["fonti_finanziamento"] = fonti

    if "pnrr" in text_lower:
        pnrr: dict = {}
        if re.search(r"(?:clausol\w+\s+social|occupazional|parità\s+di\s+genere|assunzion\w+\s+giovan)", text, re.IGNORECASE):
            pnrr["clausole_sociali"] = True
        if re.search(r"(?:DNSH|do\s+no\s+significant\s+harm|principio.*?non.*?arrecare.*?danno)", text, re.IGNORECASE):
            pnrr["DNSH"] = True
        if re.search(r"(?:tagging\s+climatico|contribut\w+\s+(?:al\s+)?(?:obiettivo|target)\s+climatico)", text, re.IGNORECASE):
            pnrr["tagging_climatico"] = True
        if pnrr:
            info_agg["condizioni_PNRR"] = pnrr

    # Collegio consultivo tecnico
    if "collegio consultivo tecnico" in text_lower:
        cont["collegio_consultivo_tecnico"] = True

    out["informazioni_aggiuntive"] = info_agg

    return out
