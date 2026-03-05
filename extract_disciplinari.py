"""
Estrattore universale di dati da disciplinari di gara.

Usa il prompt strutturato (prompt_estrazione_disciplinari.json) per estrarre
tutti i campi rilevanti da un disciplinare PDF.

Può funzionare con:
- OpenAI GPT-4 / GPT-4o
- Anthropic Claude 3.5 Sonnet / Opus
- Qualsiasi LLM compatibile con l'API OpenAI
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import pdfplumber


# ---------------------------------------------------------------------------
# 1. Carica il prompt / schema
# ---------------------------------------------------------------------------
PROMPT_PATH = Path(__file__).parent / "prompt_estrazione.json"

def load_prompt() -> dict:
    """Carica lo schema di estrazione dal file JSON."""
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 2. Estrazione testo da PDF
# ---------------------------------------------------------------------------
def _decode_cid(text: str) -> str:
    """Decodifica caratteri (cid:XX) presenti in alcuni PDF mal codificati."""
    def _repl(m):
        try:
            return chr(int(m.group(1)))
        except (ValueError, OverflowError):
            return ""
    decoded = re.sub(r"\(cid:(\d+)\)", _repl, text)
    # Pulisce spazi multipli generati dal decoding
    decoded = re.sub(r"  +", " ", decoded)
    return decoded


def extract_text_from_pdf(pdf_path: str, max_pages: int | None = None) -> str:
    """Estrae tutto il testo da un PDF disciplinare."""
    text_parts = []
    has_cid = False
    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages[:max_pages] if max_pages else pdf.pages
        for i, page in enumerate(pages):
            page_text = page.extract_text() or ""
            if "(cid:" in page_text:
                has_cid = True
                page_text = _decode_cid(page_text)
            if page_text.strip():
                text_parts.append(f"--- Pagina {i+1} ---\n{page_text}")

            # Prova anche ad estrarre tabelle
            tables = page.extract_tables()
            for table in tables:
                if table:
                    table_str = "\n".join(
                        " | ".join(str(cell or "") for cell in row)
                        for row in table
                    )
                    if "(cid:" in table_str:
                        table_str = _decode_cid(table_str)
                    text_parts.append(f"[TABELLA pag.{i+1}]\n{table_str}")

    result = "\n\n".join(text_parts)
    if has_cid:
        # Rimuove caratteri di controllo residui (spazi, newline extra)
        result = re.sub(r"\n{3,}", "\n\n", result)
    return result


# ---------------------------------------------------------------------------
# 3. Costruzione del prompt per LLM
# ---------------------------------------------------------------------------
def build_extraction_prompt(pdf_text: str) -> list[dict]:
    """Costruisce il prompt completo per l'LLM."""
    prompt_data = load_prompt()

    system_msg = prompt_data["system_prompt"]

    # Aggiungi le istruzioni di estrazione
    istruzioni = "\n".join(prompt_data["istruzioni_estrazione"])

    # Costruisci lo schema (rimuovi i campi _descrizione per brevità)
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


# ---------------------------------------------------------------------------
# 4. Chiamata LLM (supporta OpenAI e Anthropic)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 5. Utilità per parsing importi e sezioni
# ---------------------------------------------------------------------------

def _parse_euro(raw: str) -> float | None:
    """Converte una stringa di importo in float. Gestisce i formati italiani."""
    if not raw:
        return None
    s = raw.strip().replace("€", "").replace("\u20ac", "").replace("Ç", "").strip()
    # Rimuove punti iniziali ("€.138.047,15") e trailing punctuation
    s = s.lstrip(".")
    s = s.rstrip(".,;: ")
    s = re.sub(r"\s+", "", s)
    # Formato italiano: 1.554.800,00  →  1554800.00
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
    """Estrae il testo di una sezione delimitata da heading iniziale e finale.

    Supporta:
    - Match esatto case-insensitive (originale)
    - Match con numeri/punti variabili prima del heading (es. "10." → "10.1")
    - Match con spazi/newline interni (il PDF potrebbe spezzare le parole)
    """
    text_lower = full_text.lower()
    start_idx = -1

    def _is_toc_entry(pos: int) -> bool:
        """Controlla se la posizione trovata è una voce dell'Indice/TOC (con '......')."""
        after = full_text[pos:pos + 300]
        # Le voci TOC hanno puntini dopo il titolo: "TITOLO SEZIONE ......... 34"
        return bool(re.search(r"\.{5,}", after))

    for pat in start_patterns:
        pat_lower = pat.lower()
        # 1) Match esatto (salta voci TOC)
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
        # 2) Match con spazi/newline flessibili tra le parole
        words = pat_lower.split()
        if len(words) >= 2:
            flex_pat = r"\s+".join(re.escape(w) for w in words)
            for m in re.finditer(flex_pat, text_lower):
                if not _is_toc_entry(m.start()):
                    start_idx = m.start()
                    break
        if start_idx >= 0:
            break
        # 3) Match con prefisso numerico variabile (es. "10. GARANZIA" → cerca solo "GARANZIA")
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
        # Prova senza prefisso numerico
        stripped = re.sub(r"^\d+[\.\)]\s*", "", ep_lower).strip()
        if stripped and stripped != ep_lower and len(stripped) > 5:
            idx = text_lower.find(stripped, start_idx + 50)
            if idx >= 0 and idx < end_idx:
                end_idx = idx

    return full_text[start_idx : min(start_idx + max_len, end_idx)]


def _clean(s: str | None) -> str | None:
    """Pulisce spazi multipli e newline da una stringa."""
    if not s:
        return None
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None


# ---------------------------------------------------------------------------
# 6. Estrazione regole base COMPLETA (fallback senza LLM)
# ---------------------------------------------------------------------------
def extract_rules_based(text: str) -> dict:
    """
    Estrazione COMPLETA basata su regole/regex.
    Copre ~60+ campi dello schema usando pattern matching aggressivo + estrazione
    per sezioni del documento.
    """
    text_lower = text.lower()

    result = {
        "informazioni_generali": {},
        "tipo_procedura": {},
        "piattaforma_telematica": {},
        "suddivisione_lotti": {"lotti": []},
        "importi_complessivi": {},
        "tempistiche": {},
        "requisiti_partecipazione": {
            "soggetti_ammessi": {},
            "idoneita_professionale": {},
            "capacita_economico_finanziaria": {},
            "capacita_tecnico_professionale": {},
            "gruppo_di_lavoro": {"figure_professionali": []},
        },
        "criteri_valutazione": {"offerta_tecnica": {"criteri": []}, "offerta_economica": {}},
        "offerta_tecnica_formato": {},
        "garanzie": {"garanzia_provvisoria": {}, "garanzia_definitiva": {}, "polizza_RC_professionale": {}},
        "subappalto": {},
        "avvalimento": {},
        "sopralluogo": {},
        "documentazione_amministrativa": {"contributo_ANAC": {}, "imposta_bollo": {}},
        "soccorso_istruttorio": {},
        "cause_esclusione": {},
        "aggiudicazione": {"stipula_contratto": {}},
        "penali": {},
        "sicurezza": {},
        "CAM_criteri_ambientali": {},
        "controversie": {},
        "informazioni_aggiuntive": {"note": []},
    }

    ig = result["informazioni_generali"]
    tp = result["tipo_procedura"]
    pt = result["piattaforma_telematica"]
    sl = result["suddivisione_lotti"]
    ic = result["importi_complessivi"]
    temp = result["tempistiche"]
    rp = result["requisiti_partecipazione"]
    cv = result["criteri_valutazione"]
    gar = result["garanzie"]
    sub = result["subappalto"]
    avv = result["avvalimento"]
    sop = result["sopralluogo"]
    doc = result["documentazione_amministrativa"]

    # ======================================================================
    # A) INFORMAZIONI GENERALI
    # ======================================================================

    # --- Titolo / Oggetto dell'appalto ---
    # Strategia multi-livello per massimizzare il recupero
    # Cerchiamo solo nei primi 8000 car (il titolo è sempre nell'intestazione)
    _titolo_text = text[:8000]
    titolo_patterns = [
        # Livello 1: label esplicita "OGGETTO: ..." o "Oggetto dell'appalto: ..."
        r"(?:OGGETTO\s+DELL[\u2019']?\s*APPALTO|OGGETTO\s+DELLA\s+GARA|OGGETTO\s+DELL[\u2019']?\s*AFFIDAMENTO|OGGETTO)\s*[:\s\u2013\-]+\s*([^\n]+(?:\n(?!CIG|CUP|Art\.|\d+[.)\s]+[A-Z]|\s*\n)[^\n]+)*)",
        # Livello 2: "PROCEDURA ... PER ..." nell'intestazione
        r"(?:PROCEDURA\s+(?:APERTA|NEGOZIATA|RISTRETTA|COMPETITIVA)\s+(?:PER|RELATIVA\s+A)\s+(?:IL\s+|LA\s+|L[\u2019']|AL\s+|ALLA\s+)?)([^\n]+(?:\n(?!CIG|CUP|Pag\.|Disciplinare|DISCIPLINARE|Importo|\d+[.)\s]+[A-Z]|\s*\n)[^\n]+)*)",
        # Livello 3: "avente per/ad oggetto ..."
        r"avente\s+(?:per|ad)\s+oggetto\s*[:\s]+([^\n]+(?:\n(?!CIG|CUP|Art\.|\s*\n)[^\n]+)*)",
        # Livello 4: "DISCIPLINARE DI GARA PER ..."
        r"DISCIPLINARE\s+DI\s+GARA\s+(?:PER|RELATIV[OA]\s+A)\s+(?:IL\s+|LA\s+|L[\u2019'])?([^\n]+(?:\n(?!CIG|CUP|Pag\.|\s*\n)[^\n]+)*)",
        # Livello 4b: "DISCIPLINARE DI GARA" seguito da paragrafo descrittivo (senza PER)
        r"DISCIPLINARE\s+DI\s+GARA\s*\n\s*([^\n]{25,}(?:\n(?!\s*\(?CUP|\(?CIG|\d+[.)\s]+[A-Z]|\s*\n)[^\n]+)*)",
        # Livello 5: "APPALTO DI/DEI/PER ..."
        r"(?:APPALTO|AFFIDAMENTO)\s+(?:DEI|DI|DEL|PER)\s+([^\n]+(?:\n(?!CIG|CUP|Pag\.|Importo|\s*\n)[^\n]+)*)",
        # Livello 6: "BANDO DI GARA PER ..."
        r"BANDO\s+DI\s+GARA\s+(?:PER|RELATIV[OA]\s+A)\s+(?:IL\s+|LA\s+|L[\u2019'])?([^\n]+(?:\n(?!CIG|CUP|Pag\.|\s*\n)[^\n]+)*)",
        # Livello 7: "GARA ... PER ..."
        r"GARA\s+(?:(?:EUROPEA|COMUNITARIA|PUBBLICA)\s+)?(?:PER|RELATIVA)\s+(?:A\s+)?(?:IL\s+|LA\s+|L[\u2019'])?([^\n]+(?:\n(?!CIG|CUP|Art\.|\s*\n)[^\n]+)*)",
        # Livello 8: "OGGETTO: " su una riga, valore sulla successiva
        r"(?:^|\n)\s*OGGETTO\s*[:\n]\s*\n\s*([^\n]{10,500})",
        # Livello 9: articolo numerato con oggetto
        r"(?:\d+[.)\s]+)?(?:OGGETTO|Oggetto)\s+(?:dell[\u2019']?\s*)?(?:appalto|gara|affidamento|servizio|incarico)\s*[:.\-]*\n+\s*([^\n]{10,500})",
    ]
    # Parole che non dovrebbero apparire in un titolo
    _titolo_nope = {"importo", "euro ", "ribasso", "oneri sicurezza", "soggett"}
    for pat in titolo_patterns:
        m = re.search(pat, _titolo_text, re.IGNORECASE)
        if m:
            titolo = _clean(m.group(1))
            if titolo and len(titolo) > 15:
                # Tronca a frasi significative (rimuovi coda con importi/numeri)
                titolo = re.split(r"(?:Importo|CIG|CUP|Euro\s+[\d.]|Pag\.\s*\d)", titolo, flags=re.IGNORECASE)[0]
                titolo = _clean(titolo)
                if titolo and len(titolo) > 15 and not any(nw in titolo.lower() for nw in _titolo_nope):
                    ig["titolo"] = titolo[:500]
                    break

    # Fallback multi-step: prima riga sostanziosa nell'intestazione
    if "titolo" not in ig:
        # Prova con header più ampio e keyword più estese
        header_lines = text[:5000].split("\n")
        candidates = []
        for line in header_lines:
            line = line.strip()
            if len(line) < 25:
                continue
            line_lower = line.lower()
            kw_count = sum(1 for kw in [
                "servizi", "appalto", "progett", "affidamento", "procedura",
                "lavori", "ingegneria", "architettura", "direzione",
                "coordinamento", "fornitura", "manutenzione", "realizzazione",
                "costruzione", "ristrutturazione", "adeguamento", "restauro",
                "incarico", "consulenza", "assistenza"
            ] if kw in line_lower)
            if kw_count > 0:
                candidates.append((kw_count, len(line), line))
        if candidates:
            # Prendi la riga con più keyword (a parità, la più lunga)
            candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            ig["titolo"] = _clean(candidates[0][2])[:500]

    # --- Stazione appaltante ---
    sa = {}
    # Cerchiamo nei primi 5000 caratteri
    header = text[:5000]
    header_lower = header.lower()

    # Pattern per stazione appaltante
    sa_patterns = [
        r"(?:STAZIONE\s+APPALTANTE|AMMINISTRAZIONE\s+AGGIUDICATRICE)\s*[:\n]\s*(.+?)(?:\n\n|\nAtti|\nPROCEDURA|\nDISCIPLINARE|\nCIG)",
        # ATER / Azienda Territoriale
        r"(?:L['’])(ATER\s+della\s+provincia\s+di\s+\w+)",
        r"(ATER\s+[-–]\s+Azienda\s+Territoriale[^\n]+)",
        r"(Comune\s+di\s+[A-Z][A-Za-z\s]+?)(?:\s*\(|\s*\n|\s*$|\s*,)",
        r"(Citt[àa]\s+Metropolitana\s+di\s+\w+(?:\s+di\s+\w+)?)",
        r"(Provincia\s+di\s+\w+)",
        r"(?:AMMINISTRAZIONE\s+DELEGANTE\s*\n?\s*)(COMUNE\s+DI\s+[A-Z]+(?:\s*\([A-Z]+\))?)",
        r"(Centrale\s+Unica\s+di\s+Committenza\s+[^\n]+)",
    ]
    for pat in sa_patterns:
        m = re.search(pat, header, re.IGNORECASE | re.MULTILINE)
        if m:
            name = _clean(m.group(1))
            if name and len(name) > 3:
                sa["denominazione"] = name
                break

    # Tipo ente
    if sa.get("denominazione"):
        den_lower = sa["denominazione"].lower()
        if "ater" in den_lower:
            sa["tipo_ente"] = "ATER"
        elif "comune" in den_lower:
            sa["tipo_ente"] = "Comune"
        elif "città metropolitana" in den_lower or "citta metropolitana" in den_lower:
            sa["tipo_ente"] = "Città Metropolitana"
        elif "provincia" in den_lower:
            sa["tipo_ente"] = "Provincia"
        elif "asl" in den_lower or "azienda sanitaria" in den_lower:
            sa["tipo_ente"] = "ASL"
        elif "cuc" in den_lower or "centrale unica" in den_lower:
            sa["tipo_ente"] = "CUC"

    # CUC (Centrale Unica di Committenza)
    m_cuc = re.search(r"(?:CENTRALE\s+UNICA\s+DI\s+COMMITTENZA|CUC)\s*[:\-\n]\s*(.+?)(?:\n|$)", header, re.IGNORECASE)
    if m_cuc:
        cuc_name = _clean(m_cuc.group(1))
        if cuc_name and len(cuc_name) > 3:
            sa["CUC"] = cuc_name

    # PEC
    m_pec = re.search(r"(?:PEC|pec)\s*[:\s]+([a-zA-Z0-9_.+-]+@(?:pec\.)[a-zA-Z0-9-]+\.[a-zA-Z.]+)", text)
    if m_pec:
        sa["PEC"] = m_pec.group(1).lower()

    # Email
    m_email = re.search(r"(?:e-?mail|mail|Email)\s*[:\s]+([a-zA-Z0-9_.+-]+@(?!pec\.)[a-zA-Z0-9-]+\.[a-zA-Z.]+)", text, re.IGNORECASE)
    if m_email:
        sa["email"] = m_email.group(1).lower()

    # Sito web
    m_web = re.search(r"(?:sito|website|web)\s*[:\s]*(https?://[^\s\n]+)", text, re.IGNORECASE)
    if m_web:
        sa["sito_web"] = m_web.group(1)

    # Direzione/Area (skip if it's part of the title, e.g. "DIREZIONE LAVORI E")
    m_dir = re.search(r"(?:DIREZIONE|SETTORE|AREA|SERVIZIO)\s+([A-Z][^\n]{5,60})", header)
    if m_dir:
        dir_text = _clean(m_dir.group(0))
        # Non prendere se è parte del titolo dell'appalto (direzione lavori, ecc.)
        if dir_text and ig.get("titolo") and dir_text[:30].lower() not in ig["titolo"][:100].lower():
            sa["area_direzione"] = dir_text

    if sa:
        ig["stazione_appaltante"] = sa

    # --- RUP ---
    rup = {}
    _rup_false = {"responsabile", "procedimento", "progetto", "documento",
                  "informatico", "firmato", "digitale", "unico", "servizio",
                  "direzione", "settore", "procedura", "affidamento",
                  "email", "pec", "tel", "telefono", "fax", "indirizzo"}
    rup_patterns = [
        # Pattern 0: "RUP: Responsabile ..., Ing. Nome Cognome"
        r"(?:R\.?U\.?P\.?)\s*[:\s]+(?:[Rr]esponsabile[^\n]*?,\s*)?(?:(?:Dott\.?(?:ssa)?|Ing\.?|Arch\.?|Geom\.?|Prof\.?)\s+)?([A-Z][a-zàèéìòù]+(?:[ \t]+[A-Z][a-zàèéìòù]+){1,3})",
        # Pattern 1: "Responsabile ... Procedimento, Ing. Nome Cognome"
        r"[Rr]esponsabile\s+(?:[Uu]nico\s+)?(?:del\s+)?(?:[Pp]rogetto|[Pp]rocedimento|procedimento\s+e\s+del\s+progetto)[,.\s:]+(?:(?:(?:è|e)\s+)?(?:il\s+|la\s+)?)?(?:(?:Dott\.?(?:ssa)?|Ing\.?|Arch\.?|Geom\.?|Prof\.?)\s+)?([A-Z][a-zàèéìòù]+(?:[ \t]+[A-Z][a-zàèéìòù]+){1,3})",
        # Pattern 2: "R.U.P. ... Ing. Nome Cognome" con testo intermedio
        r"(?:R\.?U\.?P\.?|Responsabile\s+(?:Unico\s+)?(?:del\s+)?(?:Progetto|Procedimento)).{0,300}?(?:[Dd]ott\.?(?:ssa)?|[Ii]ng\.?|[Aa]rch\.?|[Gg]eom\.?|[Pp]rof\.?)\s+([A-Z][a-zàèéìòù]+(?:[ \t]+[A-Z][a-zàèéìòù]+){1,3})",
        # Pattern 3: "l'arch./ing. Nome Cognome"
        r"(?:R\.?U\.?P\.?|Responsabile\s+(?:Unico\s+)?(?:del\s+)?(?:Progetto|Procedimento)).{0,300}?(?:(?:è|e)\s+)?l['’]([Aa]rch|[Ii]ng|[Dd]ott)\.?\s+([A-Z][a-zàèéìòù]+(?:[ \t]+[A-Z][a-zàèéìòù]+){1,3})",
    ]
    for i, pat in enumerate(rup_patterns):
        m = re.search(pat, text, re.DOTALL)
        if m:
            candidate = _clean(m.group(2) if i == len(rup_patterns) - 1 else m.group(1))
            if candidate:
                # Rimuove parole spurie in coda (Email, Tel, Pec, ecc.)
                candidate = re.sub(r'\s+(?:Email|Pec|Tel|Telefono|Fax|Indirizzo|Documento|Firmato)\b.*', '', candidate)
                candidate = _clean(candidate)
            if candidate and not any(w in candidate.lower() for w in _rup_false):
                rup["nome"] = candidate
                break

    # Qualifica RUP
    if rup.get("nome"):
        # Pulisce nomi RUP: rimuove parole spurie (Documento, informatico, ecc.)
        rup_name = rup["nome"]
        rup_name = re.sub(r'\s+(?:Documento|informatico|sottoscritt|digitale|firmato|Responsabile)\b.*', '', rup_name)
        rup["nome"] = _clean(rup_name)
        rup_area = text[max(0, text.find(rup["nome"]) - 200):text.find(rup["nome"]) + 200]
        rup_area_lower = rup_area.lower()
        for qual in ["Dott.ssa", "Dott.", "Ing.", "Arch.", "Geom.", "Prof."]:
            if qual.lower() in rup_area_lower:
                rup["qualifica"] = qual.rstrip(".")
                break
        m_email_rup = re.search(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z.]+)", rup_area)
        if m_email_rup:
            rup["email"] = m_email_rup.group(1).lower()

    if rup:
        ig["RUP"] = rup

    # --- Numero bando ---
    m_bando = re.search(r"(?:BANDO|bando|Atti di gara)\s*(?:n\.?\s*|n°\s*)?([A-Z0-9/\-]+\d{4}/?\d*)", text)
    if m_bando:
        ig["numero_bando"] = m_bando.group(1)

    # --- Determina a contrarre ---
    det = {}
    det_patterns = [
        r"determina\w*\s+(?:\w+\s+){0,3}a\s+contrarre\s+n\.?\s*(\d+(?:/\d+)?)\s+del\s+(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})",
        r"determinazione\s+(?:\w+\s+){0,3}n\.?\s*(\d+(?:/\d+)?)\s+del\s+(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})",
        r"determina\s+a\s+contrarre\s+n\.?\s*(\d+(?:/\d+)?)\s+del\s+(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})",
    ]
    for det_pat in det_patterns:
        m_det = re.search(det_pat, text, re.IGNORECASE)
        if m_det:
            det["numero"] = m_det.group(1)
            det["data"] = m_det.group(2)
            break
    if det:
        ig["determina_a_contrarre"] = det

    # --- CIG ---
    # Standard format: CIG followed by 10+ alphanumeric chars
    cigs = re.findall(r"CIG[:\s]*([A-Za-z0-9]{10,})", text)
    if not cigs:
        # Try with spaces/separators inside: "CIG: A04F 9CC1 1A" → join
        cig_spaced = re.findall(r"CIG[:\s]*((?:[A-Za-z0-9]{2,5}\s*){3,})", text)
        for cs in cig_spaced:
            cleaned = re.sub(r'\s+', '', cs)
            if len(cleaned) >= 10:
                cigs.append(cleaned)
    if not cigs:
        # Try with "Codice CIG" or "cod. CIG"
        cigs = re.findall(r"(?:Codice|cod\.?)\s+CIG[:\s]*([A-Za-z0-9]{10,})", text, re.IGNORECASE)
    if cigs:
        unique_cigs = list(dict.fromkeys(cigs))
        ig["CIG"] = unique_cigs[0]
        if len(unique_cigs) > 1:
            ig["CIG_per_lotto"] = [
                {"lotto": i + 1, "CIG": c} for i, c in enumerate(unique_cigs)
            ]

    # --- CUP ---
    cups = re.findall(r"CUP[:\s]*([A-Z][A-Z0-9]{14})", text)
    if cups:
        unique_cups = list(dict.fromkeys(cups))
        ig["CUP"] = unique_cups[0]
        if len(unique_cups) > 1:
            ig["CUP_tutti"] = unique_cups

    # --- CUI ---
    m_cui = re.search(r"CUI[:\s]*([\d]+)", text)
    if m_cui:
        ig["CUI"] = m_cui.group(1)

    # --- Codice NUTS ---
    nuts_matches = re.findall(r"NUTS[:\s]*([A-Z]{2}[A-Z0-9]{1,4})", text)
    if nuts_matches:
        ig["codice_NUTS"] = nuts_matches[0]

    # --- CPV ---
    cpv_matches = re.findall(r"(\d{8}-\d)", text)
    if cpv_matches:
        unique_cpv = list(dict.fromkeys(cpv_matches))
        ig["CPV_principale"] = unique_cpv[0]
        if len(unique_cpv) > 1:
            ig["CPV_secondari"] = unique_cpv[1:]

    # --- Finanziamento ---
    fin = {}
    fin_patterns = [
        (r"(?:PNRR|Piano Nazionale di Ripresa e Resilienza)", "PNRR"),
        (r"Fondi?\s+(?:di\s+)?[Bb]ilancio", "Fondi di Bilancio"),
        (r"(?:fondi?|finanziament\w+)\s+(?:region\w+|POR|FSC|FESR)", "Fondi regionali/europei"),
        (r"Fondo\s+Complementare", "Fondo Complementare"),
    ]
    for pat, label in fin_patterns:
        if re.search(pat, text, re.IGNORECASE):
            fin["fonte"] = label
            break
    if fin:
        ig["finanziamento"] = fin

    # ======================================================================
    # B) TIPO PROCEDURA
    # ======================================================================

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

    # Riferimento normativo
    m_ref = re.search(r"(?:ai sensi|ex)\s+(art\.?\s*\d+[^.]{0,80}D\.?Lgs\.?\s*(?:n\.?\s*)?\d+/\d{4})", text, re.IGNORECASE)
    if m_ref:
        tp["riferimento_normativo"] = _clean(m_ref.group(1))

    # Inversione procedimentale
    if "inversione procedimentale" in text_lower or "art. 107 comma 3" in text_lower or "articolo 107, comma 3" in text_lower:
        tp["inversione_procedimentale"] = True

    # Accordo Quadro
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

    # Concessione
    if "concessione" in text_lower and ("art. 182" in text_lower or "art. 194" in text_lower):
        tp["concessione"] = True

    # ======================================================================
    # C) PIATTAFORMA TELEMATICA
    # ======================================================================

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
        # Usa word boundary per evitare falsi positivi (es. 'mepa' in 'HomePage')
        if re.search(r'\b' + re.escape(key) + r'\b', text_lower):
            pt["nome"] = nome
            if gestore:
                pt["gestore"] = gestore
            break

    # URL piattaforma
    m_url = re.search(r"(?:https?://[^\s\n]+(?:aria|sintel|tuttogare|net4market|traspare|mepa|acquistinrete|start\.toscana|empulia|intercent|maggioli|asmecomm|appalti|gare)[^\s\n]*)", text, re.IGNORECASE)
    if m_url:
        pt["url"] = m_url.group(0).rstrip(".,;)")
    elif not pt.get("url"):
        # Fallback: any URL near "piattaforma" or "telematica"
        m_url_gen = re.search(r"(?:piattaforma|telematica)[^.]{0,200}?(https?://[^\s\n]+)", text, re.IGNORECASE)
        if m_url_gen:
            pt["url"] = m_url_gen.group(1).rstrip(".,;)")

    # ======================================================================
    # D) LOTTI E IMPORTI
    # ======================================================================

    # Numero lotti
    lotti_nums = re.findall(r"lotto\s+(?:n\.?\s*)?(\d+)", text, re.IGNORECASE)
    if lotti_nums:
        n_lotti = max(int(n) for n in lotti_nums)
        # Sanity: se troviamo "lotto 15" in un documento di 30 pagine, è plausibile,
        # ma se troviamo cifre altissime (es. lotto 100), è probabilmente un errore
        if n_lotti > 50:
            n_lotti = 1
        sl["numero_lotti"] = n_lotti
    else:
        sl["numero_lotti"] = 1
        if "lotto unico" in text_lower or "unico lotto" in text_lower or "non suddivisa in lotti" in text_lower:
            sl["lotto_unico_motivazione"] = "Lotto unico come da disciplinare"

    # Importo totale complessivo
    imp_patterns = [
        # "importo globale della procedura è pari a € X"
        r"importo\s+globale[^€Ç\d]{0,100}?(?:pari\s+ad?\s+)?(?:€|Ç|Euro|euro|EUR)\s*\.?\s*([\d.,]+)",
        # Più specifici: "importo complessivo dell'appalto ... Euro/€ X"
        r"importo\s+complessivo\s+dell['\u2019\s]\s*appalto[^€Ç\d]{0,80}?(?:€|Ç|Euro|euro|EUR)\s*\.?\s*([\d.,]+)",
        r"importo\s+complessivo\s+(?:Euro|€|di\s+€)\s*([\d.,]+)",
        r"importo\s+complessivo[^€Ç\d]{0,40}?(?:Euro|euro|EUR)\s*\.?\s*([\d.,]+)",
        r"importo\s+complessivo[^€Ç\d]{0,80}?(?:pari\s+ad?|ammonta\s+a)\s*(?:€|Ç|Euro|euro|EUR)\s*\.?\s*([\d.,]+)",
        r"importo\s+complessivo[^€Ç\d]{0,120}?(?:di|pari\s+a|per)\s+(?:€|Ç|Euro|euro|EUR)\s*\.?\s*([\d.,]+)",
        # "Base di gara" / "importo a base di gara" — con contesto pari/ammonta
        r"(?:base\s+di\s+gara|base\s+d['\u2019]asta)[^€Ç\d]{0,60}?(?:pari\s+ad?|ammonta\s+a)\s*(?:€|Ç|Euro|euro|EUR)\s*\.?\s*([\d.,]+)",
        r"importo\s+(?:a\s+)?base\s+(?:di\s+|d['\u2019])?(?:gara|asta)\s*[€Ç:]\s*\.?\s*([\d.,]+)",
        # "Importo a base di gara ... Euro X" (solo se non preceduto da %, garanzia)
        r"(?:^|[\n.])(?:[^%\n]{0,20})importo\s+(?:a\s+)?base\s+(?:di\s+|d['\u2019])?(?:gara|asta)[^€Ç\d]{0,40}?(?:Euro|euro|EUR)\s*\.?\s*([\d.,]+)",
        # Valore concessione / STIMA TOTALE
        r"STIMA\s+TOTALE\s+DEL\s+VALORE[^€Ç\d]{0,80}?(?:€|Ç|Euro|euro|EUR)\s*\.?\s*([\d.,]+)",
        r"valore\s+(?:del(?:la)?\s+)?(?:contratto\s+di\s+)?concessione[^€Ç\d]{0,80}?(?:€|Ç|euro|Euro)\s*\.?\s*([\d.,]+)",
        r"valore\s+(?:del(?:la)?\s+)?concessione[^€Ç\d]{0,80}?(?:pari\s+a|determinat\w+\s+in)\s*[:\s]*(?:€|Ç|euro|Euro)\s*\.?\s*([\d.,]+)",
        r"(?:determinat\w+\s+in|pari\s+a)\s*[:\s]*euro\s+([\d.,]+(?://\d{2})?)",
        # Pattern generici
        r"importo\s+totale\s+(?:stimat\w+\s+)?(?:dell['\u2019]appalto\s+)?(?:€|Euro|:)?\s*(?:pari\s+a\s+)?(?:€\s*)?([\d.,]+)",
        r"valore\s+globale\s+dell['\u2019]?\s*appalto[^\d]{0,60}?([\d.,]+)",
        r"valore\s+massimo\s+stimato[^\d]{0,80}?(?:Euro|€)\s*([\d.,]+)",
        r"importo\s+complessivo[^€Ç\d]{0,80}?[€Ç]\s*\.?\s*([\d.,]+)",
        r"importo\s+dell['\u2019]?\s*appalto\s*\n?\s*(?:€|Euro)\s*([\d.,]+)",
        r"importo\s+dell['\u2019]?\s*appalto\s*[:\s]*(?:€|Euro)\s*([\d.,]+)",
        r"(?:fissato|stabilito)\s+in\s*(?:€|Ç|Euro|euro)\s*\.?\s*([\d.,]+)",
        r"TOTALE\s*[€Ç]\s*\.?\s*([\d.,]+)",
        # Numero + "euro/EUR" dopo keywords importo (cattura retroattiva)
        r"(?:importo\s+complessivo|importo\s+totale|base\s+di\s+gara|base\s+d['\u2019]asta)[^.]{0,100}?([\d.,]+)\s*(?:€|Euro|euro|EUR)",
    ]
    for pat in imp_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            raw_val = m.group(1)
            # Gestisce formato "29.905.519//00" → rimuove //00
            raw_val = re.sub(r'//\d{2}$', '', raw_val)
            val = _parse_euro(raw_val)
            if val and val > 100:
                ic["importo_totale_gara"] = val
                break

    # Importo soggetto a ribasso (deve contenere €/Euro/euro e valore > 100)
    rib_patterns = [
        r"soggett\w+\s+a\s+ribasso[^\n]{0,60}?[€]\s*([\d.,]+)",
        r"soggett\w+\s+a\s+ribasso[^\n]{0,60}?(?:Euro|euro|EUR)\s*([\d.,]+)",
        r"soggett\w+\s+a\s+ribasso\s*(?:\([^)]*\))?\s*[:\n]\s*[€]?\s*([\d.,]+)",
        r"soggett\w+\s+a\s+ribasso[^\n]{0,80}?([\d.,]+)\s*(?:€|Euro|euro|EUR)",
        r"ribasso[^.]{0,80}?(?:€|Euro|euro|EUR)\s*\.?\s*([\d.,]+)",
    ]
    for pat in rib_patterns:
        m_rib = re.search(pat, text, re.IGNORECASE)
        if m_rib:
            val = _parse_euro(m_rib.group(1))
            if val and val > 100:
                ic["importo_totale_soggetto_ribasso"] = val
                break

    # Oneri sicurezza
    oneri_patterns = [
        r"oneri\s+(?:per\s+la\s+)?sicurezza[^€Ç\d]{0,50}[€Ç\s]*([\d.,]+)",
        r"oneri\s+(?:per\s+la\s+)?sicurezza[^€Ç\d]{0,60}?(?:Euro|euro|EUR)\s*\.?\s*([\d.,]+)",
        r"oneri\s+(?:della\s+)?sicurezza[^€Ç\d]{0,60}?(?:€|Ç|Euro|euro|EUR)\s*\.?\s*([\d.,]+)",
        r"costi?\s+(?:della\s+|per\s+la\s+)?sicurezza[^€Ç\d]{0,60}?(?:€|Ç|Euro|euro|EUR)\s*\.?\s*([\d.,]+)",
    ]
    for on_pat in oneri_patterns:
        m_on = re.search(on_pat, text, re.IGNORECASE)
        if m_on:
            val = _parse_euro(m_on.group(1))
            if val:
                ic["oneri_sicurezza"] = val
                break

    # Importo lavori
    m_lav = re.search(r"importo\s+(?:dei\s+)?lavori[^€Ç\d]{0,50}[€Ç\s]*([\d.,]+)", text, re.IGNORECASE)
    if m_lav:
        val = _parse_euro(m_lav.group(1))
        if val and val > 100:
            ic["importo_lavori_complessivo"] = val

    # Quota ribassabile percentuale
    m_quota = re.search(r"(?:quota\s+)?(?:soggett\w+\s+a\s+)?ribasso[^.]{0,80}?(\d{1,3})\s*%", text, re.IGNORECASE)
    if m_quota:
        perc = int(m_quota.group(1))
        if 1 <= perc <= 100:
            ic["quota_ribassabile_percentuale"] = perc

    # Anticipazione
    m_ant = re.search(r"anticipazione[^.]{0,100}?(\d{1,3})\s*%", text, re.IGNORECASE)
    if m_ant:
        ic["anticipazione"] = {"prevista": True, "percentuale": int(m_ant.group(1))}

    # Revisione prezzi
    if "revisione" in text_lower and "prezzi" in text_lower:
        rev = {"ammessa": True}
        m_rev_perc = re.search(r"(?:revisione[^.]{0,200}?)(\d{1,3})\s*%\s*(?:dell['\u2019])?(?:variazione|incremento)", text, re.IGNORECASE)
        if not m_rev_perc:
            m_rev_perc = re.search(r"(?:soglia|superiore)\s+(?:al\s+)?(\d{1,3})\s*%", text[text_lower.find("revisione"):text_lower.find("revisione")+500], re.IGNORECASE)
        if m_rev_perc:
            rev["soglia_percentuale"] = int(m_rev_perc.group(1))
        ic["revisione_prezzi"] = rev

    # --- Dettaglio per singoli lotti ---
    # Cerchiamo blocchi "Lotto n. X" o "LOTTO X"  con importi
    for lotto_n in range(1, sl["numero_lotti"] + 1):
        lotto_data = {"numero": lotto_n}

        # Sezione del lotto — cerca la prima occorrenza "sostanziale"
        # (skippa menzioni brevi come "(Lotto 1)" nell'intro)
        lotto_patterns = [
            f"lotto\\s*(?:n\\.?\\s*)?{lotto_n}\\b",
            f"LOTTO\\s*{lotto_n}\\b",
        ]
        lotto_start = -1
        for lp in lotto_patterns:
            search_from = 0
            while True:
                m = re.search(lp, text[search_from:], re.IGNORECASE)
                if not m:
                    break
                candidate = search_from + m.start()
                # Calcola la lunghezza fino al prossimo lotto o sezione
                next_lotto = re.search(
                    f"(?:lotto\\s*(?:n\\.?\\s*)?{lotto_n + 1}\\b|^\\d+\\.\\s+[A-Z])",
                    text[candidate + 20:],
                    re.IGNORECASE | re.MULTILINE,
                )
                segment_len = next_lotto.start() + 20 if next_lotto else 8000
                if segment_len >= 200:
                    lotto_start = candidate
                    break
                # Troppo corto — probabilmente menzione breve, prova la successiva
                search_from = candidate + len(m.group(0))
            if lotto_start >= 0:
                break

        if lotto_start >= 0:
            # Trova il prossimo lotto o sezione
            next_lotto = re.search(
                f"(?:lotto\\s*(?:n\\.?\\s*)?{lotto_n + 1}\\b|^\\d+\\.\\s+[A-Z])",
                text[lotto_start + 20:],
                re.IGNORECASE | re.MULTILINE,
            )
            lotto_end = lotto_start + (next_lotto.start() + 20 if next_lotto else 8000)
            lotto_text = text[lotto_start:lotto_end]

            # Descrizione lotto (prima riga)
            first_lines = lotto_text[:500]
            m_desc = re.search(r"(?:lotto\s*(?:n\.?\s*)?\d+)\s*[:\-\n]\s*(.+?)(?:\n\n|\n(?:Tabella|IMPORTO|Prestazione))", first_lines, re.IGNORECASE | re.DOTALL)
            if m_desc:
                lotto_data["denominazione"] = _clean(m_desc.group(1))

            # Importi nel lotto
            imp_lotto = re.search(r"(?:IMPORTO\s+LOTTO|importo\s+(?:del\s+)?lotto)[^€Ç\d]{0,30}[€Ç\s]*([\d.,]+)", lotto_text, re.IGNORECASE)
            if imp_lotto:
                v = _parse_euro(imp_lotto.group(1))
                if v:
                    lotto_data["importo_base_asta"] = v

            # Importo totale base gara nel lotto
            imp_base = re.search(r"[Ii]mporto\s+totale\s+a\s+base\s+di\s+gara\s*[:\n]\s*[€Ç]?\s*([\d.,]+)", lotto_text)
            if imp_base:
                v = _parse_euro(imp_base.group(1))
                if v:
                    lotto_data["importo_base_asta"] = v

            # "Valore appalto € X" / "Valore appalto complessivo € X"
            if "importo_base_asta" not in lotto_data:
                imp_valore = re.search(
                    r"[Vv]alore\s+appalto\s*(?:complessivo)?\s*€\s*([\d.,]+)",
                    lotto_text, re.IGNORECASE,
                )
                if imp_valore:
                    v = _parse_euro(imp_valore.group(1))
                    if v and v > 100:
                        lotto_data["importo_base_asta"] = v

            # "importo contrattuale complessivo pari a € X"
            if "importo_base_asta" not in lotto_data:
                imp_contr = re.search(
                    r"importo\s+contrattuale\s+(?:complessivo\s+)?pari\s+a\s*€?\s*([\d.,]+)",
                    lotto_text, re.IGNORECASE,
                )
                if imp_contr:
                    v = _parse_euro(imp_contr.group(1))
                    if v and v > 100:
                        lotto_data["importo_base_asta"] = v

            # Importo soggetto a ribasso nel lotto
            m_rib_l = re.search(r"soggett\w+\s+a\s+ribasso[^€Ç\d]{0,50}[€Ç\s]*([\d.,]+)", lotto_text, re.IGNORECASE)
            if m_rib_l:
                v = _parse_euro(m_rib_l.group(1))
                if v:
                    lotto_data["importo_soggetto_ribasso"] = v

            # Percentuale ribasso
            m_perc = re.search(r"(\d{1,3})\s*%\s*(?:di\s+[A-Z]|del\s+corrispettivo)", lotto_text, re.IGNORECASE)
            if m_perc:
                p = int(m_perc.group(1))
                if 1 <= p <= 100:
                    lotto_data["quota_ribassabile_percentuale"] = p

            # Oneri previdenziali
            m_prev = re.search(r"[Oo]neri\s+previdenziali[^€Ç\d]{0,50}[€Ç\s]*([\d.,]+)", lotto_text)
            if m_prev:
                v = _parse_euro(m_prev.group(1))
                if v:
                    lotto_data["contributo_previdenziale"] = {"importo": v}
            m_prev_p = re.search(r"(?:CNPAIA|INARCASSA|contributo\s+previdenziale)[^.]{0,50}?(\d+(?:[.,]\d+)?)\s*%", lotto_text, re.IGNORECASE)
            if m_prev_p:
                if "contributo_previdenziale" not in lotto_data:
                    lotto_data["contributo_previdenziale"] = {}
                lotto_data["contributo_previdenziale"]["percentuale"] = float(m_prev_p.group(1).replace(",", "."))

            # Prestazioni (CPV in tabella)
            prestazioni = []
            prest_matches = re.findall(
                r"(Servizi\s+di\s+[^\n]{5,80}|Direzione\s+lavori|Coordinamento[^\n]{5,60}|"
                r"Collaudo[^\n]{5,60}|Servizi\s+tecnici[^\n]{0,60}|"
                r"Progettazione[^\n]{5,60}|Verifica[^\n]{5,60}|Indagini[^\n]{5,60})"
                r"\s+(\d{8}-\d)\s+([PS])\s+([\d.,]+)",
                lotto_text,
                re.IGNORECASE,
            )
            for desc, cpv_code, pors, imp_str in prest_matches:
                v = _parse_euro(imp_str)
                if v:
                    prestazioni.append({
                        "descrizione": _clean(desc),
                        "codice_CPV": cpv_code,
                        "tipo": "principale" if pors.upper() == "P" else "secondaria",
                        "importo": v,
                    })
            if prestazioni:
                lotto_data["prestazioni"] = prestazioni

            # Categorie opere nel lotto
            cat_matches = re.findall(
                r"(E\d{2}|S\d{2}|IA\d{2}|D\.?\d{2}|V\.?\d{2})\s+"
                r"([^\n]{10,200}?)\s+"
                r"(\d+[.,]\d{1,2})\s+"
                r"([\d.,]+(?:\s*€)?)",
                lotto_text,
            )
            categorie = []
            for cat_id, desc, complessita, imp_str in cat_matches:
                v = _parse_euro(imp_str)
                cat_entry = {
                    "id_categoria": cat_id[:1] + "." + cat_id[1:] if "." not in cat_id else cat_id,
                    "descrizione": _clean(desc)[:200],
                    "grado_complessita": float(complessita.replace(",", ".")),
                }
                if v:
                    cat_entry["importo_opera"] = v
                categorie.append(cat_entry)
            if categorie:
                lotto_data["categorie_opere"] = categorie

            # Durata
            m_dur = re.search(r"durat\w+[^.]{0,80}?(\d+)\s*(?:giorni|gg)", lotto_text, re.IGNORECASE)
            if m_dur:
                lotto_data["durata_esecuzione"] = {"giorni": int(m_dur.group(1))}
            else:
                m_dur = re.search(r"durat\w+[^.]{0,80}?(\d+)\s*mesi", lotto_text, re.IGNORECASE)
                if m_dur:
                    lotto_data["durata_esecuzione"] = {"mesi": int(m_dur.group(1))}

        sl["lotti"].append(lotto_data)

    # --- Vincoli partecipazione lotti ---
    if sl["numero_lotti"] > 1:
        sl["offerta_identica"] = bool(re.search(
            r'(?:medesima|stessa)\s+offerta[^.]{0,40}identica|offerta\s+identica[^.]{0,40}(?:entrambi|tutti)',
            text, re.IGNORECASE,
        ))
        sl["medesima_forma_giuridica"] = bool(re.search(
            r'medesima\s+forma\s+giuridica', text, re.IGNORECASE,
        ))

    # --- Fallback globale per importi per-lotto mancanti ---
    for lotto in sl["lotti"]:
        if "importo_base_asta" not in lotto:
            n = lotto["numero"]
            # "per il Lotto N, l'importo contrattuale complessivo pari a € X"
            m_ic = re.search(
                rf"Lotto\s*(?:n\.?\s*)?{n}\b.{{0,200}}?"
                rf"(?:importo\s+contrattuale\s+(?:complessivo\s+)?pari\s+a|Valore\s+appalto(?:\s+complessivo)?)"
                rf"\s*€?\s*([\d.,]+)",
                text, re.IGNORECASE | re.DOTALL,
            )
            if m_ic:
                v = _parse_euro(m_ic.group(1))
                if v and v > 100:
                    lotto["importo_base_asta"] = v
        # Denominazione fallback dalla menzione "Lotto N – Edificio/Descrizione"
        if "denominazione" not in lotto:
            n = lotto["numero"]
            m_den = re.search(
                rf"Lotto\s*(?:n\.?\s*)?{n}\s*[–\-\u2013]\s*(.{{10,200}}?)(?:\n|ID\s|CATEGORIA|\d{{1,2}}\.\d)",
                text, re.IGNORECASE,
            )
            if m_den:
                lotto["denominazione"] = _clean(m_den.group(1))

    # Se lotto unico e non abbiamo trovato importi lotto specifici, usa quelli globali
    if sl["numero_lotti"] == 1 and sl["lotti"]:
        lotto = sl["lotti"][0]
        if "importo_base_asta" not in lotto and ic.get("importo_totale_gara"):
            lotto["importo_base_asta"] = ic["importo_totale_gara"]
        # Per lotto unico, usa anche gli importi di ribasso e denominazione se disponibili
        if "importo_soggetto_ribasso" not in lotto and ic.get("importo_totale_soggetto_ribasso"):
            lotto["importo_soggetto_ribasso"] = ic["importo_totale_soggetto_ribasso"]
        if "denominazione" not in lotto and ig.get("titolo"):
            lotto["denominazione"] = ig["titolo"]
    elif sl["numero_lotti"] == 1 and not sl["lotti"]:
        # Nessun lotto trovato esplicitamente, crea uno da dati globali
        lotto = {"numero": 1}
        if ic.get("importo_totale_gara"):
            lotto["importo_base_asta"] = ic["importo_totale_gara"]
        if ig.get("titolo"):
            lotto["denominazione"] = ig["titolo"]
        sl["lotti"].append(lotto)

    # --- Estrazione lotti da tabella multi-colonna (formato QTE) ---
    # Cerca sezione tabellare con righe aventi almeno 2 importi in €/Ç
    # Per ogni riga, il penultimo importo è il "QTE netto IVA"
    _euro_sym_pat = r"[€Ç]"
    # Pattern che cattura sia numeri con separatore (3 389 899,90) sia senza (2409022,00)
    _amt_line_pat = (
        r"((?:\d{1,3}[\s.])*\d{1,3}(?:[.,]\d{1,2})?|\d{4,}(?:[.,]\d{1,2})?)\s*"
        + _euro_sym_pat
    )

    # Cerca la zona QTE / tabella importi
    qte_area = ""
    for _qte_kw in ["QTE", "quadro tecnico economico", "IMPORTO DEI LAVORI", "importo QTE"]:
        _qte_idx = text_lower.find(_qte_kw.lower())
        if _qte_idx >= 0:
            qte_area = text[max(0, _qte_idx - 200):_qte_idx + 6000]
            break

    if qte_area:
        lotti_from_qte = []
        for _qline in qte_area.split("\n"):
            _stripped = _qline.strip()
            # Salta righe di TOTALE e header di sezioni tabella
            if _stripped.upper().startswith("TOTALE"):
                continue
            if _stripped.startswith("[TABELLA"):
                continue
            _amounts_raw = re.findall(_amt_line_pat, _qline)
            if len(_amounts_raw) >= 2:
                # Penultimo importo = QTE netto
                _netto_raw = _amounts_raw[-2].replace(" ", "")
                _v_netto = _parse_euro(_netto_raw)
                if _v_netto and _v_netto > 50000:
                    lotti_from_qte.append(_v_netto)

        # Deduplica importi identici (stessi valori dal testo pagina + tabella)
        _deduped = []
        for _v in lotti_from_qte:
            if not any(abs(_v - _ex) < 1.0 for _ex in _deduped):
                _deduped.append(_v)
        lotti_from_qte = _deduped

        if len(lotti_from_qte) >= 2:
            sl["lotti"] = []
            for _li, _imp in enumerate(lotti_from_qte):
                sl["lotti"].append({"numero": _li + 1, "importo_base_asta": _imp})
            sl["numero_lotti"] = len(lotti_from_qte)

    # Anche se non abbiamo trovato categorie per lotto, cerchiamole globalmente
    all_cats = re.findall(
        r"((?:Edilizia|Strutture?|Impianti|Idraulica|Viabilit|Infrastruttur)\w*)\s+"
        r"(E\.?\d{2}|S\.?\d{2}|IA\.?\d{2}|D\.?\d{2}|V\.?\d{2})\s+"
        r"([^\n]{10,250}?)\s+"
        r"(\d+[.,]\d{1,2})\s+"
        r"([\d.,]+(?:\s*€)?)",
        text,
    )
    if all_cats:
        categorie_globali = []
        for cat_type, cat_id, desc, complessita, imp_str in all_cats:
            v = _parse_euro(imp_str)
            entry = {
                "tipo_opera": _clean(cat_type),
                "id_categoria": cat_id[:1] + "." + cat_id[1:] if "." not in cat_id and len(cat_id) == 3 else cat_id,
                "descrizione": _clean(desc)[:200],
                "grado_complessita": float(complessita.replace(",", ".")),
            }
            if v:
                entry["importo_opera"] = v
            categorie_globali.append(entry)
        if categorie_globali:
            ic["categorie_opere_dettaglio"] = categorie_globali

    # Cerca anche categorie in formato semplice (senza tabella) - include E/S/IA/D/V
    cats_simple = re.findall(r"(?:categori\w+|class\w+)\s+((?:[ESDIV]\.?\d{2}(?:\s*[-,e]\s*)?)+)", text, re.IGNORECASE)
    if cats_simple:
        found_cats = set()
        for match in cats_simple:
            for c in re.findall(r"[ESDIV]\.?\d{2}", match):
                found_cats.add(c[:1] + "." + c[-2:] if "." not in c else c)
        ig["categorie_trovate"] = sorted(found_cats)

    # Categorie opere OG/OS (per lavori, tipico in concessioni)
    og_os_matches = re.findall(r"\b(O[GS]\d{1,2})\b", text)
    if og_os_matches:
        unique_og_os = list(dict.fromkeys(og_os_matches))
        if "categorie_trovate" not in ig:
            ig["categorie_trovate"] = []
        for cat in unique_og_os:
            if cat not in ig["categorie_trovate"]:
                ig["categorie_trovate"].append(cat)

    # ======================================================================
    # E) TEMPISTICHE
    # ======================================================================

    # Scadenza offerte
    scad_patterns = [
        # "entro e non oltre le ore 12,00 del giorno 22.12.2025"
        r"(?:entro\s+(?:e\s+non\s+oltre\s+)?(?:le\s+)?ore\s+)(\d{1,2}[:.,:]\d{2})\s+del\s+(?:giorno\s+)?(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})",
        # "entro il giorno DD/MM/YYYY ore HH:MM"
        r"(?:entro\s+il\s+(?:giorno\s+)?)(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})\s+(?:ore\s+)?(\d{1,2}[:.]\d{2})",
        # "termine/scadenza...DD/MM/YYYY ore HH:MM"
        r"(?:scadenza|termine)[^.]{0,200}?(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})\s+(?:ore\s+)?(\d{1,2}[:.]\d{2})",
        # "presentazione offerte...DD/MM/YYYY ore HH:MM" (senza prefisso scadenza)
        r"presentazione\s+(?:dell['\u2019]\s*)?offert\w+[^.]{0,150}?(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})\s+(?:ore\s+)?(\d{1,2}[:.]\d{2})",
        # Standard: "scadenza/termine presentazione offerte ...data"
        r"(?:scadenza|termine)\s+(?:per\s+la\s+)?(?:presentazione|ricezione)\s+(?:dell['\u2019]\s*)?offert\w+[^.]{0,100}?(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})",
        # "le offerte dovranno pervenire entro ... DD/MM/YYYY"
        r"offert\w+\s+(?:dovranno|devono)\s+(?:essere\s+)?(?:presentat\w+|trasmess\w+|inviat\w+|pervenire)[^.]{0,200}?(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})(?:\s+(?:ore\s+)?(\d{1,2}[:.]\d{2}))?",
        # "ore HH:MM del DD mese YYYY" (formato data scritta italiana)
        r"(?:entro\s+(?:e\s+non\s+oltre\s+)?(?:le\s+)?ore\s+)(\d{1,2}[:.,:]\d{2})\s+del\s+(\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4})",
        # Tabella/campo con solo data e ora vicino a "scadenza"  
        r"(?:scadenza|termine)[^.]{0,60}?(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})",
    ]
    for sp in scad_patterns:
        m_scad = re.search(sp, text, re.IGNORECASE | re.DOTALL)
        if m_scad:
            groups = m_scad.groups()
            if len(groups) >= 2 and groups[1]:
                # Determina quale gruppo è data e quale è orario
                g1, g2 = groups[0], groups[1]
                if re.match(r"\d{1,2}[:.,:]\d{2}$", g1) and re.match(r"\d{1,2}[/.\-]", g2):
                    # g1=ore, g2=data
                    temp["scadenza_offerte"] = f"{g2} ore {g1}"
                else:
                    # g1=data, g2=ore
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

    # Apertura buste / prima seduta
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

    # Durata contratto / concessione (anni o mesi)
    dur_c = {}
    # Durata in anni
    m_dur_anni = re.search(
        r"(?:durat\w+\s+(?:del(?:la)?\s+)?(?:contratto|concessione|appalto|servizio|incarico)|"
        r"(?:gestione|opera)\s+[eè]\s+concess\w+\s+per)"
        r"[^.]{0,120}?(\d+)\s*ann[io]",
        text, re.IGNORECASE,
    )
    if m_dur_anni:
        dur_c["anni"] = int(m_dur_anni.group(1))

    # Durata in giorni
    m_dur_gen = re.search(
        r"durat\w+\s+(?:complessiv\w+\s+)?(?:dell?['\u2019]\s*)?(?:contratto|appalto|servizio|incarico|lotto|procedura|prestazion\w+)"
        r"[^.]{0,150}?(\d+)\s*(?:giorni|gg)\s+(?:naturali\s+e\s+consecutivi|natural\w+|lavora\w+|solari\w+)?",
        text, re.IGNORECASE,
    )
    if not m_dur_gen:
        # Fallback più ampio: "durata complessiva ... stimata in N giorni"
        m_dur_gen = re.search(
            r"durat\w+\s+complessiv\w+[^.]{0,200}?(?:stimat\w+\s+in|pari\s+a|di)\s*(\d+)\s*(?:giorni|gg)",
            text, re.IGNORECASE,
        )
    if m_dur_gen:
        dur_c["giorni"] = int(m_dur_gen.group(1))
        temp["durata_esecuzione_giorni"] = dur_c["giorni"]

    # Durata in mesi
    m_dur_m = re.search(r"durat\w+\s+(?:dell?['\u2019]\s*)?(?:contratto|appalto|servizio)[^.]{0,100}?(\d+)\s*mesi", text, re.IGNORECASE)
    if m_dur_m:
        dur_c["mesi"] = int(m_dur_m.group(1))
        temp["durata_esecuzione_mesi"] = dur_c["mesi"]

    # Fasi lavori/gestione
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
        result["durata_contratto"] = dur_c

    # Società di scopo (per concessioni PPP)
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
        if "durata_contratto" not in result:
            result["durata_contratto"] = {}
        result["durata_contratto"]["societa_di_scopo"] = sds

    # Termine stipula
    m_stip = re.search(r"stipula[^.]{0,100}?(\d+)\s*giorni", text, re.IGNORECASE)
    if m_stip:
        temp["termine_stipula_giorni"] = int(m_stip.group(1))

    # Standstill
    m_stand = re.search(r"stand\s*[-]?\s*still[^.]{0,80}?(\d+)\s*giorni", text, re.IGNORECASE)
    if m_stand:
        temp["standstill_giorni"] = int(m_stand.group(1))

    # ======================================================================
    # F) REQUISITI DI PARTECIPAZIONE
    # ======================================================================

    req_section = _section_text(
        text,
        ["REQUISITI DI ORDINE SPECIALE", "REQUISITI DI ORDINE GENERALE", "REQUISITI DI PARTECIPAZIONE", "6. REQUISITI"],
        ["7. AVVALIMENTO", "8. SUBAPPALTO", "AVVALIMENTO", "SUBAPPALTO", "GARANZI"],
        max_len=25000,
    )

    # Soggetti ammessi
    sogg = rp["soggetti_ammessi"]
    sogg_types = []
    sogg_map = {
        "professionisti singoli": "professionisti singoli",
        "società di professionisti": "società di professionisti",
        "società di ingegneria": "società di ingegneria",
        "raggruppamenti temporanei": "raggruppamenti temporanei",
        "consorzi stabili": "consorzi stabili",
        "geie": "GEIE",
        "aggregazioni di rete": "aggregazioni di rete",
        "studi associati": "studi associati",
    }
    for key, label in sogg_map.items():
        if key in text_lower:
            sogg_types.append(label)
    if sogg_types:
        sogg["tipologie"] = sogg_types

    # Giovane professionista
    if "giovane professionista" in text_lower:
        sogg["obbligo_giovane_professionista"] = True
        m_giov = re.search(r"giovane\s+professionista[^.]{0,200}", text, re.IGNORECASE)
        if m_giov:
            frag = m_giov.group(0)
            m_def = re.search(r"(laureat\w+[^.]{0,100}(?:anni?|mesi))", frag, re.IGNORECASE)
            if m_def:
                sogg["definizione_giovane"] = _clean(m_def.group(1))

    # Fatturato globale
    fat_glob = rp["capacita_economico_finanziaria"]
    m_fat = re.search(
        r"fatturato\s+globale[^.]{0,200}?(?:€|Euro)\s*([\d.,]+)|"
        r"fatturato\s+globale[^.]{0,200}?(?:pari\s+al?\s+)?(\w+\s+dell['\u2019]\s*importo)",
        text, re.IGNORECASE,
    )
    if m_fat:
        fat_glob["fatturato_globale"] = {"richiesto": True}
        if m_fat.group(1):
            v = _parse_euro(m_fat.group(1))
            if v:
                fat_glob["fatturato_globale"]["importo_minimo"] = v
        if m_fat.group(2):
            fat_glob["fatturato_globale"]["rapporto_con_importo_gara"] = _clean(m_fat.group(2))
    
    m_fat_per = re.search(r"fatturato\s+globale[^.]{0,200}?((?:ultimo|miglio)\w*\s+\w+\s+(?:anni?|esercizi?)[^.]{0,50})", text, re.IGNORECASE)
    if m_fat_per:
        if "fatturato_globale" not in fat_glob:
            fat_glob["fatturato_globale"] = {"richiesto": True}
        fat_glob["fatturato_globale"]["periodo_riferimento"] = _clean(m_fat_per.group(1))

    # Servizi analoghi
    srv = rp["capacita_tecnico_professionale"]
    m_srv = re.search(
        r"servizi\s+(?:analoghi|di\s+punta)[^.]{0,200}?((?:ultimo|ultimi)\s+\w+\s*\w*)",
        text, re.IGNORECASE,
    )
    if m_srv:
        srv["servizi_analoghi"] = {
            "richiesti": True,
            "periodo_riferimento": _clean(m_srv.group(1)),
        }

    # Requisiti categorie per servizi analoghi
    cat_req = re.findall(
        r"(?:categori\w+|class\w+)\s+(E\.?\d{2}|S\.?\d{2}|IA\.?\d{2}|D\.?\d{2}|V\.?\d{2})[^.]{0,100}?"
        r"(?:import\w+[^€\d]{0,30}[€\s]*([\d.,]+)|class\w+\s+(\w+))",
        req_section, re.IGNORECASE,
    )
    if cat_req:
        cats_list = []
        for cat_id, imp_val, classe in cat_req:
            entry = {"categoria": cat_id[:1] + "." + cat_id[-2:] if "." not in cat_id else cat_id}
            if imp_val:
                v = _parse_euro(imp_val)
                if v:
                    entry["importo_minimo"] = v
            if classe:
                entry["classe_minima"] = classe
            cats_list.append(entry)
        if cats_list:
            if "servizi_analoghi" not in srv:
                srv["servizi_analoghi"] = {"richiesti": True}
            srv["servizi_analoghi"]["categorie_richieste"] = cats_list

    # Personale tecnico medio
    m_pers = re.search(r"(?:organico|personale)\s+(?:tecnico\s+)?medio[^.]{0,100}?(\d+)\s*(?:unit[àa]|dipendent)", text, re.IGNORECASE)
    if m_pers:
        srv["personale_tecnico_medio"] = {
            "richiesto": True,
            "numero_minimo": int(m_pers.group(1)),
        }

    # Gruppo di lavoro - figure professionali
    gdl = rp["gruppo_di_lavoro"]
    ruoli_patterns = [
        (r"(?:progettist\w+\s+(?:struttur\w+|architetton\w+|impiantist\w+))", None),
        (r"(?:direttore\s+(?:dei\s+)?lavori)", None),
        (r"(?:coordinatore\s+(?:per\s+la\s+)?sicurezza\s+in\s+fase\s+di\s+(?:progettazione|esecuzione))", None),
        (r"(?:geologo)", None),
        (r"(?:collaudator\w+\s+(?:struttur\w+|static\w+|tecnico\s+amministrativ\w+)?)", None),
        (r"(?:direttore\s+operativo)", None),
        (r"(?:ispettore\s+di\s+cantiere)", None),
        (r"(?:responsabile\s+(?:della\s+)?integrazione)", None),
        (r"(?:professionista\s+antincendio)", None),
        (r"(?:tecnico\s+competente\s+in\s+acustica)", None),
        (r"(?:coordinatore\s+del\s+gruppo)", None),
        (r"(?:coordinatore\s+della\s+progettazione)", None),
        (r"(?:BIM\s+manager|BIM\s+coordinator|BIM\s+specialist)", None),
        (r"(?:esperto\s+ambientale|esperto\s+CAM)", None),
        (r"(?:topografo)", None),
        # "Tecnico esperto in/di ..." (generico, cattura ruoli specialistici)
        (r"(?:tecnico\s+esperto\s+(?:in|di)\s+[^\n]{5,80}?)(?=\n|\s{2,}|Laurea)", None),
    ]
    figure = []
    for pat, _ in ruoli_patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        for m in matches:
            role = _clean(m)
            if not role:
                continue
            # Dedup: controlla corrispondenza esatta, prefisso comune (primi 40 car) o sottostringhe
            rl = role.lower()
            is_dup = any(
                rl == ex or rl[:40] == ex[:40] or rl.startswith(ex[:40]) or ex.startswith(rl[:40])
                or rl in ex or ex in rl
                for ex in (f.get("ruolo", "").lower() for f in figure)
            )
            if is_dup:
                continue
            entry = {"ruolo": role}
            # Cerca requisiti specifici vicino alla menzione
            idx_r = text_lower.find(role.lower())
            if idx_r >= 0:
                ctx = text[idx_r:idx_r + 300]
                m_req_r = re.search(r"(?:abilitazione|iscrizione|iscritt\w+)\s+([^.]{10,100})", ctx, re.IGNORECASE)
                if m_req_r:
                    entry["requisiti"] = _clean(m_req_r.group(0))
            entry["_pos"] = idx_r
            figure.append(entry)

    # Post-dedup: remove shorter roles that appear within the description of a longer role
    if len(figure) > 1:
        to_remove = set()
        for i in range(len(figure)):
            pi = figure[i].get("_pos", -1)
            ri = figure[i].get("ruolo", "").lower()
            if pi < 0:
                continue
            for j in range(len(figure)):
                if i == j:
                    continue
                pj = figure[j].get("_pos", -1)
                rj_len = len(figure[j].get("ruolo", ""))
                if pj < 0:
                    continue
                # Use a window around the longer role's full description block
                ctx_end = min(len(text_lower), pj + max(rj_len, 200) + 200)
                if len(ri) < rj_len and pi >= pj and pi < ctx_end:
                    if ri in text_lower[pj:ctx_end]:
                        to_remove.add(i)
                        break
        figure = [f for idx, f in enumerate(figure) if idx not in to_remove]
    for f in figure:
        f.pop("_pos", None)

    if figure:
        gdl["figure_professionali"] = figure
    else:
        # Fallback: estrai figure da liste numerate/puntate nella sezione requisiti
        # Pattern: "a) ... b) ..." or "1) ... 2) ..." or "- ... - ..."
        fig_section = req_section or _section_text(
            text,
            ["GRUPPO DI LAVORO", "FIGURE PROFESSIONALI", "COMPOSIZIONE DEL GRUPPO", "STRUTTURA OPERATIVA",
             "TEAM DI PROGETTAZIONE", "SOGGETTI RICHIESTI"],
            ["AVVALIMENTO", "SUBAPPALTO", "GARANZI", "CRITERI", "9.", "10."],
            max_len=10000,
        )
        if fig_section:
            # Pattern per liste con ruoli professionali
            fig_list_patterns = [
                # "n. 1 <ruolo>" oppure "n° 1 <ruolo>"
                r"n\.?\s*°?\s*(\d+)\s+((?:[A-Z][a-z]+\s+){1,5}[a-z]+(?:\s+[a-z]+){0,3})",
                # "1) <Ruolo con descrizione>"  o "a) <ruolo>"
                r"(?:^|\n)\s*(?:\d+|[a-h])\s*[.)]\s*([A-Z][^\n]{10,120}?)(?:\n|;)",
                # "- <ruolo> (requisiti)" o "• <ruolo>"
                r"(?:^|\n)\s*[-•–]\s*([A-Z][^\n]{10,120}?)(?:\n|;)",
                # Tabella numerata: "1 ... Tecnico esperto ... 2 ... Tecnico ..."
                r"(?:^|\n)\s*(\d)\s+(Tecnico\s+[^\n]{10,120}?)(?=\s+Laurea|\s+Diploma|\s+Abilit|\n\n)",
            ]
            for fig_pat in fig_list_patterns:
                fig_matches = re.findall(fig_pat, fig_section, re.MULTILINE)
                if fig_matches and len(fig_matches) >= 2:
                    for fm in fig_matches:
                        role_text = fm[-1] if isinstance(fm, tuple) else fm
                        role_text = _clean(role_text)
                        if role_text and len(role_text) > 5:
                            # Filtra voci che non sono ruoli professionali
                            role_lower = role_text.lower()
                            if any(kw in role_lower for kw in [
                                "progettist", "direttore", "coordinat", "ingegner",
                                "architet", "geolog", "collaudat", "ispettor",
                                "responsabil", "tecnico", "professionista", "esperto",
                                "specialista", "consulente", "verificat", "topograf",
                                "bim", "operativ", "lavori", "sicurezza", "struttur",
                                "impianti", "edilizia", "manutenzione",
                            ]):
                                figure.append({"ruolo": role_text[:200]})
                    if figure:
                        gdl["figure_professionali"] = figure
                        break

    # Cumulabilità ruoli (stessa persona può coprire più ruoli)
    cumul_patterns = [
        r"(?:stesso\s+soggetto|medesimo\s+professionista|stessa\s+persona)[^.]{0,100}?(?:più\s+ruoli|più\s+funzioni|più\s+incarichi|cumular)",
        r"(?:cumulabil\w+|cumulo)[^.]{0,100}?(?:ruol\w+|incari\w+|funzion\w+|prestazion\w+)",
        r"(?:ruol\w+|incari\w+|funzion\w+)[^.]{0,100}?(?:cumulabil\w+|cumulo|sovrappo\w+|coincider\w+)",
        r"(?:più\s+ruoli|più\s+funzioni)\s+(?:possono\s+)?(?:essere\s+)?(?:affidati|ricoperti|svolti)\s+(?:da|dal)\s+(?:un\s+)?(?:unico|stesso|medesimo)",
    ]
    cumul_found = False
    for cp in cumul_patterns:
        if re.search(cp, text, re.IGNORECASE):
            cumul_found = True
            break
    # Anche il contrario: "non cumulabili"
    non_cumul = re.search(
        r"(?:non\s+(?:sono\s+)?cumulabil\w+|non\s+(?:è\s+)?(?:consentit\w+|ammess\w+)\s+il\s+cumulo|"
        r"incompatibil\w+[^.]{0,60}?ruol\w+)",
        text, re.IGNORECASE,
    )
    if cumul_found or non_cumul:
        gdl["ruoli_cumulabili"] = cumul_found and not non_cumul

    # ======================================================================
    # G) CRITERI DI VALUTAZIONE
    # ======================================================================

    crit_section = _section_text(
        text,
        ["CRITERIO DI AGGIUDICAZIONE", "CRITERI DI VALUTAZIONE", "18. CRITERIO", "5.1 Criteri"],
        ["SVOLGIMENTO DELLE OPERAZIONI DI GARA", "23.", "24.", "25.", "19."],
        max_len=20000,
    )

    ot = cv["offerta_tecnica"]
    oe = cv["offerta_economica"]

    # Punteggio offerta tecnica (cerchiamo PRIMA nel crit_section, poi globale)
    pt_patterns = [
        r"offerta\s+tecnica\s*(?:[:=]|\(|punti\s+)?\s*(?:max\.?\s*|massimo\s+)?(?:punti\s+)?(\d{1,3})\s*(?:punti|pt|punto|/100|\))",
        r"(?:Punteggio\s+)?[Oo]fferta\s+tecnica[^\d]{0,30}(\d{1,3})\s*(?:punti|pt|/)",
        # Pattern semplice "Offerta tecnica 60" (senza punteggiatura intermedia)
        r"[Oo]fferta\s+[Tt]ecnica\s+(\d{1,3})\s*(?:$|\n)",
        r"(?:tecnic\w+|qualit[àa])\s*[:=]\s*(?:max\.?\s*)?(?:punti\s+)?(\d{1,3})",
        r"(\d{1,3})\s*punti\s*[-–]\s*offerta\s+tecnica",
    ]
    for pt_pat in pt_patterns:
        m_pt = re.search(pt_pat, crit_section or text, re.IGNORECASE | re.MULTILINE)
        if m_pt:
            val_pt = int(m_pt.group(1))
            if 5 <= val_pt <= 100:
                ot["punteggio_massimo"] = val_pt
                break

    # Punteggio offerta economica
    pe_patterns = [
        r"offerta\s+economica\s*(?:[:=]|\(|punti\s+)?\s*(?:max\.?\s*|massimo\s+)?(?:punti\s+)?(\d{1,3})\s*(?:punti|pt|punto|/100|\))",
        r"(?:Punteggio\s+)?[Oo]fferta\s+economica[^\d]{0,30}(\d{1,3})\s*(?:punti|pt|/)",
        # Pattern semplice "Offerta economica 40"
        r"[Oo]fferta\s+[Ee]conomica\s+(\d{1,3})\s*(?:$|\n)",
        r"(?:economic\w+|prezzo)\s*[:=]\s*(?:max\.?\s*)?(?:punti\s+)?(\d{1,3})",
        r"(\d{1,3})\s*punti\s*[-–]\s*offerta\s+economica",
    ]
    for pe_pat in pe_patterns:
        m_pe = re.search(pe_pat, crit_section or text, re.IGNORECASE)
        if m_pe:
            val_pe = int(m_pe.group(1))
            if 1 <= val_pe <= 100:
                oe["punteggio_massimo"] = val_pe
                break

    # Soglia sbarramento
    soglia_patterns = [
        r"soglia\s+(?:di\s+)?sbarramento[^0-9]{0,50}?(\d{1,3})",
        # SOGLIA MINIMA ... TOTALE DI PUNTI N (solo pattern con TOTALE)
        r"SOGLIA\s+MINIMA.{0,500}?TOTALE\s+(?:DI\s+)?PUNTI\s+(\d{1,3})",
        r"soglia\s+minima.{0,500}?totale\s+(?:di\s+)?punti\s+(\d{1,3})",
        r"al\s+di\s*sotto\s+(?:della\s+)?soglia[^0-9]{0,50}?(\d{1,3})",
    ]
    for soglia_pat in soglia_patterns:
        m_soglia = re.search(soglia_pat, text, re.DOTALL)
        if m_soglia:
            for g in m_soglia.groups():
                if g:
                    val_s = int(g)
                    if 5 <= val_s <= 100:
                        ot["soglia_sbarramento"] = val_s
                        break
            if "soglia_sbarramento" in ot:
                break

    # Riparametrazione
    if "riparametrazione" in text_lower or "riparametrati" in text_lower:
        ot["riparametrazione"] = True

    # Sub-criteri valutazione (pattern tabellare comune)
    # Pattern: codice (A, B, C, 1, 2... o A.1, B.2...) + descrizione + punti
    sub_criteri = re.findall(
        r"(?:^|\n)\s*([A-Z](?:\.\d{1,2})?|\d{1,2}(?:\.\d{1,2})?)\s*[.\-\)\s]\s*"
        r"(.{10,200}?)\s+"
        r"(\d{1,3}(?:[.,]\d{1,2})?)\s*(?:punti|pt|punto|\s+\d|\n|$)",
        crit_section,
        re.IGNORECASE | re.MULTILINE,
    )

    # Supplemento: righe compatte "CODE PUNTI" senza descrizione (es. "A.2 20")
    compact_criteri = re.findall(
        r"(?:^|\n)\s*([A-Z](?:\.\d{1,2})?)\s+(\d{1,3})\s*$",
        crit_section,
        re.MULTILINE,
    )
    existing_codes = {s[0] for s in sub_criteri}
    for code, pts in compact_criteri:
        if code not in existing_codes:
            sub_criteri.append((code, code, pts))  # usa il codice come descrizione provvisoria

    # Supplemento: criteri da TABELLA pdfplumber (colonne con |)
    table_sections = re.findall(
        r"\[TABELLA[^\]]*\]\n(.*?)(?=\n\n|\n---|\n\[TABELLA|\Z)", crit_section, re.DOTALL
    )
    existing_codes = {s[0] for s in sub_criteri}
    for tab in table_sections:
        for row in tab.split("\n"):
            cells = [c.strip() for c in row.split("|")]
            cells = [c for c in cells if c]
            if not cells:
                continue
            codes = [c for c in cells if re.match(r"^[A-Z](?:\.\d{1,2})?$", c)]
            numbers = [c for c in cells if re.match(r"^\d{1,3}$", c) and 1 <= int(c) <= 100]
            descs = [c for c in cells if len(c) > 5 and not re.match(r"^[\d,.]+$", c)
                     and not re.match(r"^[A-Z](?:\.\d)?$", c)]
            for code in codes:
                if code in existing_codes:
                    continue
                desc = descs[0] if descs else code
                pts = numbers[0] if numbers else None
                if pts:
                    sub_criteri.append((code, desc, pts))
                    existing_codes.add(code)

    # Arricchimento descrizioni: cerca nomi criteri dalla tabella per codici con desc corte
    _table_descs = {}
    def _pick_desc_after_code(cells, code):
        """Return the best desc cell that appears AFTER code in the row."""
        try:
            code_idx = cells.index(code)
        except ValueError:
            return None
        # Look for desc cells after the code position
        for c in cells[code_idx + 1:]:
            if len(c) > 5 and not re.match(r"^[\d,.]+$", c) and not re.match(r"^[A-Z](?:\.\d{1,2})?$", c):
                return c
        # Fallback: any desc cell in the row
        for c in cells:
            if c != code and len(c) > 8 and not re.match(r"^[\d,.]+$", c) and not re.match(r"^[A-Z](?:\.\d{1,2})?$", c):
                return c
        return None

    for tab in table_sections:
        for row in tab.split("\n"):
            cells = [c.strip() for c in row.split("|")]
            cells = [c for c in cells if c]
            codes = [c for c in cells if re.match(r"^[A-Z](?:\.\d{1,2})?$", c)]
            for code in codes:
                if code not in _table_descs:
                    desc = _pick_desc_after_code(cells, code)
                    if desc:
                        _table_descs[code] = desc
    # Anche parse pipe-delimited rows senza marker [TABELLA]
    for row in crit_section.split("\n"):
        if "|" not in row:
            continue
        cells = [c.strip().strip("*") for c in row.split("|")]
        cells = [c.strip() for c in cells if c.strip()]
        codes = [c for c in cells if re.match(r"^[A-Z](?:\.\d{1,2})?$", c)]
        for code in codes:
            if code not in _table_descs:
                desc = _pick_desc_after_code(cells, code)
                if desc:
                    _table_descs[code] = desc
    # Supplemento: nome da intestazione bold "**A.1 – Competenze..."
    for m_bold in re.finditer(
        r"\*\*\s*([A-Z](?:\.\d{1,2})?)\s*[\u2013\u2014\-–—]\s*(.{10,120}?)\s*\*\*",
        crit_section,
    ):
        code_b = m_bold.group(1)
        desc_b = m_bold.group(2).strip()
        if code_b not in _table_descs:
            _table_descs[code_b] = desc_b
    # Nomi noti per criteri standard
    _known_names = {
        "A": "Qualità e adeguatezza del gruppo di lavoro proposto",
        "B": "Promozione dell'inserimento di giovani professionisti",
        "C": "Approccio metodologico, organizzativo e coordinamento tecnico",
        "D": "Tecniche di rilievo e restituzione grafica",
        "E": "Certificazioni",
    }
    # Aggiorna sub_criteri con descrizioni migliori
    enriched = []
    for codice, desc, punti_str in sub_criteri:
        table_desc = _table_descs.get(codice, "")
        is_sub = "." in codice
        # Per sub-criteri: preferisci la descrizione da tabella (intestazione autorevole)
        # Per criteri principali: preferisci la più lunga
        if is_sub and table_desc and len(table_desc) >= 10:
            better_desc = table_desc
        elif table_desc and len(table_desc) > len(desc):
            better_desc = table_desc
        else:
            better_desc = desc
        # Se la descrizione è troppo corta, prova nel testo vicino al codice
        if len(better_desc) < 10:
            m_name = re.search(
                rf"(?:^|\n)\s*{re.escape(codice)}\s+([A-Z][^\d\n]{{8,120}})",
                crit_section, re.MULTILINE,
            )
            if m_name:
                better_desc = m_name.group(1).strip()
        # Per criteri principali: usa _known_names (nomi autorevoli e completi)
        if codice in _known_names:
            better_desc = _known_names[codice]
        # Ultima risorsa: usa il codice stesso
        if not better_desc or len(better_desc) < 3:
            better_desc = codice
        enriched.append((codice, better_desc, punti_str))
    sub_criteri = enriched
    criteri_parsed = []
    for codice, desc, punti_str in sub_criteri:
        try:
            punti = float(punti_str.replace(",", "."))
        except ValueError:
            continue
        if punti <= 0 or punti > 100:
            continue
        desc_clean = _clean(desc)
        # Rimuovi caratteri private-use (U+F02D ecc.) che appaiono come bullet/dash nei PDF
        if desc_clean:
            desc_clean = re.sub(r'^[\uf02d\uf0b7\uf0a7\uf020-\uf0ff\s\-–—]+', '', desc_clean).strip()
        if desc_clean and len(desc_clean) > 5:
            is_sub = "." in codice
            criteri_parsed.append({
                "codice": codice,
                "nome": desc_clean[:300],
                "punteggio": punti,
                "livello": "sub_criterio" if is_sub else "criterio",
            })

    # Filtra rumore (entries dal TOC, numeri di pagina, ecc.)
    criteri_filtered = [
        c for c in criteri_parsed
        if not re.search(r'(CRITERI DI VALUTAZIONE|METODO DI|CALCOLO DEI|PROCEDURA|OFFERTA TECNICA\s*\.)', c['nome'], re.IGNORECASE)
        and '...' not in c['nome']
        and len(c['nome']) > 8
    ]
    if criteri_filtered:
        ot["criteri"] = criteri_filtered

    # Modalità offerta economica
    if "ribasso percentuale" in text_lower:
        oe["modalita_offerta"] = "ribasso_percentuale"
    elif "prezzo" in text_lower and "offerta economica" in text_lower:
        oe["modalita_offerta"] = "prezzo"

    # Cifre decimali
    m_dec = re.search(r"(\d+)\s*cifre\s+decimali", text, re.IGNORECASE)
    if m_dec:
        oe["cifre_decimali"] = int(m_dec.group(1))

    # Formula economica
    m_form = re.search(r"(?:formula|metodo)[^.]{0,30}?(?:bilineare|non lineare|lineare|proporzionale)", text, re.IGNORECASE)
    if m_form:
        oe["formula"] = _clean(m_form.group(0))

    # Verifica anomalia
    va = {}
    if "anomal" in text_lower or "anormalmente bass" in text_lower:
        va["prevista"] = True
        m_va = re.search(r"(?:art\.?\s*110|art\.?\s*97)", text, re.IGNORECASE)
        if m_va:
            va["riferimento_normativo"] = _clean(m_va.group(0))
    if va:
        cv["verifica_anomalia"] = va

    # ======================================================================
    # H) FORMATO OFFERTA TECNICA
    # ======================================================================

    otf = result["offerta_tecnica_formato"]
    m_pag = re.search(r"(?:max|massimo|fino a)\s*(\d+)\s*pagin\w+", text, re.IGNORECASE)
    if m_pag:
        otf["pagine_massime"] = int(m_pag.group(1))

    m_char = re.search(r"(?:Times\s+New\s+Roman|Arial|Calibri|Garamond)\s*(\d+)", text, re.IGNORECASE)
    if m_char:
        otf["carattere"] = _clean(m_char.group(0))

    m_inter = re.search(r"interlinea\s*[:\s]*([\d.,]+)", text, re.IGNORECASE)
    if m_inter:
        otf["interlinea"] = m_inter.group(1)

    # ======================================================================
    # I) GARANZIE
    # ======================================================================

    gp = gar["garanzia_provvisoria"]
    gd = gar["garanzia_definitiva"]
    pol = gar["polizza_RC_professionale"]

    # Garanzia provvisoria
    gar_section = _section_text(text, ["GARANZIA PROVVISORIA", "10. GARANZI"], ["GARANZIA DEFINITIVA", "11.", "SOPRALLUOGO"], max_len=5000)

    _gar_check = gar_section.lower() if gar_section.strip() else text_lower
    _gar_non_dovuta = (
        "non dovuta" in _gar_check
        or "non è dovuta" in _gar_check
        or "non richiesta" in _gar_check
        or "non è richiesta" in _gar_check
    )
    # Fallback: check full text for explicit negation about garanzia provvisoria
    if not _gar_non_dovuta and not gar_section.strip():
        _gar_non_dovuta = bool(re.search(
            r"non\s+è\s+richiest\w+\s+la\s+garanzia\s+provvisoria",
            text, re.IGNORECASE,
        ))
    if _gar_non_dovuta:
        gp["dovuta"] = False
    else:
        # Pattern per percentuale garanzia: "2% dell'importo", "pari al 2%", ecc.
        gp_perc_patterns = [
            r"(\d+(?:[.,]\d+)?)\s*%\s*(?:dell?\s*['\u2019]?\s*import)",
            r"pari\s+(?:al?\s+)?(\d+(?:[.,]\d+)?)\s*%",
            r"misura\s+(?:del?\s+)?(\d+(?:[.,]\d+)?)\s*%",
        ]
        for _gpp in gp_perc_patterns:
            m_gp = re.search(_gpp, gar_section, re.IGNORECASE)
            if m_gp:
                _gpval = float(m_gp.group(1).replace(",", "."))
                if 0.5 <= _gpval <= 10:
                    gp["dovuta"] = True
                    gp["percentuale"] = _gpval
                    break
        if "percentuale" not in gp and re.search(r"garanzia\s+provvisoria", text, re.IGNORECASE):
            gp["dovuta"] = True

    # Importo garanzia provvisoria - cerca sia € che "euro"
    gp_imp_patterns = [
        r"(?:garanzia\s+provvisoria|cauzione)[^€\d]{0,200}?(?:importo\s*)?(?:pari\s+ad?\s+)?(?:€|euro)\s*\.?\s*([\d.,]+)",
        r"(?:garanzia\s+provvisoria|cauzione)[^€\d]{0,80}?[€\s]*([\d.,]+)",
        r"(?:importo|pari\s+ad?)\s+(?:€|euro)\s*\.?\s*([\d.,]+)",
    ]
    for gp_pat in gp_imp_patterns:
        m_gp_imp = re.search(gp_pat, gar_section, re.IGNORECASE | re.DOTALL)
        if m_gp_imp:
            v = _parse_euro(m_gp_imp.group(1))
            if v and v > 100:
                gp["importo"] = v
                break

    # Garanzia definitiva
    gd_section = _section_text(text, ["GARANZIA DEFINITIVA"], ["11.", "SOPRALLUOGO", "12.", "PAGAMENTO", "CONTRIBUTO"], max_len=3000)
    m_gd = re.search(r"(\d+(?:[.,]\d+)?)\s*%\s*(?:dell?\s*['\u2019]?\s*import)", gd_section, re.IGNORECASE)
    if m_gd:
        gd["dovuta"] = True
        gd["percentuale"] = float(m_gd.group(1).replace(",", "."))

    m_gd_forma = re.search(r"(?:cauzione|fideiussione|garanzia fideiussoria)", gd_section, re.IGNORECASE)
    if m_gd_forma:
        gd["forma"] = _clean(m_gd_forma.group(0))

    # Polizza RC
    if "polizza" in text_lower and ("responsabilità civile" in text_lower or "rc professionale" in text_lower or "errori e omissioni" in text_lower):
        pol["richiesta"] = True
        m_pol = re.search(r"polizza[^.]{0,200}?(?:errori\s+e\s+omissioni|responsabilità\s+civile)[^.]{0,200}", text, re.IGNORECASE)
        if m_pol:
            pol["copertura"] = _clean(m_pol.group(0))[:200]

    # ======================================================================
    # J) SUBAPPALTO
    # ======================================================================

    sub_section = _section_text(text, ["SUBAPPALTO"], ["9.", "REQUISITI DI PARTECIPAZIONE", "GARANZI", "SOPRALLUOGO"], max_len=5000)
    if sub_section:
        sub["ammesso"] = True
        m_sub_perc = re.search(r"(\d+)\s*%\s*(?:dell?\s*['\u2019]?\s*import|del\s+valore|del\s+contratto)", sub_section, re.IGNORECASE)
        if m_sub_perc:
            sub["limite_percentuale"] = int(m_sub_perc.group(1))
        if "non è ammesso" in sub_section.lower() or "non ammesso" in sub_section.lower():
            sub["ammesso"] = False
    elif "subappalto" in text_lower:
        sub["ammesso"] = True

    # ======================================================================
    # K) AVVALIMENTO
    # ======================================================================

    avv_section = _section_text(text, ["AVVALIMENTO"], ["8. SUBAPPALTO", "SUBAPPALTO", "REQUISITI", "GARANZI"], max_len=3000)
    if avv_section:
        avv["ammesso"] = True
        if "non è ammesso" in avv_section.lower() or "non ammesso" in avv_section.lower():
            avv["ammesso"] = False

    # ======================================================================
    # L) SOPRALLUOGO
    # ======================================================================

    sop_section = _section_text(text, ["SOPRALLUOGO"], ["12.", "PAGAMENTO", "CONTRIBUTO", "MODALITÀ", "13."], max_len=3000)
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

        # Modalita sopralluogo
        m_sop_mod = re.search(
            r"(?:sopralluogo)[^.]{0,200}?(?:da\s+effettuar\w+|si\s+svol\w+|mediante|con\s+modalit\w+)\s+([^.]{10,200})",
            sop_section, re.IGNORECASE | re.DOTALL,
        )
        if m_sop_mod:
            sop["modalita"] = _clean(m_sop_mod.group(1))[:200]

        # Contatti prenotazione
        m_sop_cont = re.search(
            r"(?:prenotazion\w+|appuntamento|contattare|richied\w+)[^.]{0,100}?([\w.]+@[\w.]+\.\w{2,}|\d{2,4}[/-]?\d{4,})",
            sop_section, re.IGNORECASE,
        )
        if m_sop_cont:
            sop["contatti_prenotazione"] = _clean(m_sop_cont.group(1))

        # Termine sopralluogo
        m_sop_term = re.search(
            r"sopralluogo[^.]{0,200}?entro\s+(?:il\s+)?(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})",
            sop_section, re.IGNORECASE | re.DOTALL,
        )
        if m_sop_term:
            sop["termine"] = m_sop_term.group(1)

    # ======================================================================
    # M) DOCUMENTAZIONE AMMINISTRATIVA
    # ======================================================================

    # Contributo ANAC
    anac = doc["contributo_ANAC"]
    anac_patterns = [
        r"contributo\s+(?:a\s+favore\s+dell?\s*['\u2019]?\s*)?ANAC[^€\d]{0,100}?[€\s]*([\d.,]+)",
        # "contributo ... Anticorruzione per un importo pari a euro 560,00"
        r"contributo\s+(?:previsto\s+)?(?:dalla\s+legge\s+)?(?:in\s+favore\s+)?(?:dell?\s*['\u2019]?\s*)?(?:Autorit[àa]\s+Nazionale\s*\n?\s*Anticorruzione|ANAC)[^€\d]{0,200}?(?:€|euro)\s*\.?\s*([\d.,]+)",
        r"(?:€|euro)\s*\.?\s*([\d.,]+)[^\n]{0,50}?(?:contributo|ANAC)",
        r"contributo\s+ANAC[^\d]{0,60}?([\d.,]+)",
    ]
    for anac_pat in anac_patterns:
        m_anac = re.search(anac_pat, text, re.IGNORECASE | re.DOTALL)
        if m_anac:
            v = _parse_euro(m_anac.group(1))
            if v and v >= 20:  # min ANAC contributo is ~20€
                anac["dovuto"] = True
                anac["importo_totale"] = v
                break
    if "importo_totale" not in anac and "contributo" in text_lower and "anac" in text_lower:
        anac["dovuto"] = True

    # Importi ANAC per lotto
    anac_lotti = re.findall(r"lotto\s+\d[^€]{0,30}?[€\s]*([\d.,]+)\s*(?:€|\n)", text, re.IGNORECASE)
    if anac_lotti and len(anac_lotti) >= 2:
        anac["importi_per_lotto"] = [
            {"lotto": i + 1, "importo": _parse_euro(v)}
            for i, v in enumerate(anac_lotti)
            if _parse_euro(v)
        ]

    # Imposta di bollo
    bollo = doc["imposta_bollo"]
    m_bollo = re.search(r"(?:imposta\s+di\s+bollo|bollo)[^€\d]{0,50}?[€\s]*([\d.,]+)", text, re.IGNORECASE)
    if m_bollo:
        v = _parse_euro(m_bollo.group(1))
        if v:
            bollo["importo"] = v

    # DGUE
    if "dgue" in text_lower or "espd" in text_lower:
        doc["DGUE"] = {"richiesto": True}
        if "firma digitale" in text_lower:
            doc["DGUE"]["firma"] = "digitale"

    # Soccorso istruttorio
    si = result["soccorso_istruttorio"]
    if "soccorso istruttorio" in text_lower:
        si["ammesso"] = True
        # Cerca "termine di N giorni" vicino a "soccorso istruttorio"
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

    # ======================================================================
    # N) CAUSE DI ESCLUSIONE
    # ======================================================================

    ce = result["cause_esclusione"]
    if "art. 94" in text or "art.94" in text or "articolo 94" in text_lower:
        ce["automatiche"] = {"riferimento": "art. 94 D.Lgs. 36/2023"}
    if "art. 95" in text or "art.95" in text or "articolo 95" in text_lower:
        ce["non_automatiche"] = {"riferimento": "art. 95 D.Lgs. 36/2023"}
    if "self cleaning" in text_lower or "self-cleaning" in text_lower or "art. 96" in text:
        ce["self_cleaning"] = {"ammesso": True, "riferimento": "art. 96 D.Lgs. 36/2023"}

    # ======================================================================
    # O) AGGIUDICAZIONE E STIPULA
    # ======================================================================

    agg = result["aggiudicazione"]
    m_max_lotti = re.search(r"(?:massimo|max)\s+(\d+)\s+lott[oi]", text, re.IGNORECASE)
    if m_max_lotti:
        agg["numero_lotti_massimi_per_concorrente"] = int(m_max_lotti.group(1))
    elif "un solo lotto" in text_lower or "aggiudicatario di un solo lotto" in text_lower:
        agg["numero_lotti_massimi_per_concorrente"] = 1

    stip = agg["stipula_contratto"]
    if "atto pubblico" in text_lower:
        stip["forma"] = "atto_pubblico"
    elif "scrittura privata" in text_lower:
        stip["forma"] = "scrittura_privata"
    elif "forma pubblica amministrativa" in text_lower:
        stip["forma"] = "forma_pubblica_amministrativa"

    m_stand = re.search(r"stand\s*-?\s*still[^.]{0,50}?(\d+)\s*giorni", text, re.IGNORECASE)
    if not m_stand:
        m_stand = re.search(r"(\d+)\s*giorni[^.]{0,30}comunicazione[^.]{0,30}aggiudicazione", text, re.IGNORECASE)
    if m_stand:
        stip["termine_standstill_giorni"] = int(m_stand.group(1))

    # ======================================================================
    # P) PENALI
    # ======================================================================

    pen = result["penali"]
    m_pen = re.search(r"penal[ei][^.]{0,200}?(\d+(?:[.,]\d+)?)\s*%\s*(?:(?:al\s+)?giorn|per\s+ogni\s+giorn)", text, re.IGNORECASE)
    if m_pen:
        pen["previste"] = True
        pen["percentuale_giornaliera"] = float(m_pen.group(1).replace(",", "."))
    m_pen_max = re.search(r"penal[ei][^.]{0,300}?(?:massim\w+|tetto|limit\w+)[^.]{0,50}?(\d+(?:[.,]\d+)?)\s*%", text, re.IGNORECASE)
    if m_pen_max:
        pen["tetto_massimo_percentuale"] = float(m_pen_max.group(1).replace(",", "."))

    # ======================================================================
    # Q) SICUREZZA
    # ======================================================================

    sic = result["sicurezza"]
    if "natura intellettuale" in text_lower:
        sic["DUVRI"] = {"richiesto": False, "nota": "Servizio di natura intellettuale, nessun rischio interferenza"}
        result["informazioni_aggiuntive"]["natura_servizio"] = "natura intellettuale"
    elif re.search(r"non\s+(?:sussiste|è\s+richiest\w+|è\s+necessari\w+|previsto)\b[^.]{0,60}duvri|duvri[^.]{0,60}non\s+(?:richiest\w+|necessari\w+|previst\w+|dovut\w+)", text, re.IGNORECASE):
        sic["DUVRI"] = {"richiesto": False, "nota": "DUVRI non richiesto come da disciplinare"}
    elif "duvri" in text_lower:
        sic["DUVRI"] = {"richiesto": True}

    # ======================================================================
    # R) CAM
    # ======================================================================

    cam = result["CAM_criteri_ambientali"]
    if "criteri ambientali minimi" in text_lower or "cam" in text_lower:
        cam["applicabili"] = True
        m_cam = re.search(r"(?:decreto|D\.?M\.?)\s*(?:n\.?\s*)?(\d+)\s+del\s+(\d{1,2}\s+\w+\s+\d{4})", text, re.IGNORECASE)
        if m_cam:
            cam["decreto_riferimento"] = f"D.M. n. {m_cam.group(1)} del {m_cam.group(2)}"

    # ======================================================================
    # S) CONTROVERSIE
    # ======================================================================

    cont = result["controversie"]
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

    # ======================================================================
    # T) COMMISSIONE GIUDICATRICE
    # ======================================================================

    info_agg = result["informazioni_aggiuntive"]
    m_comm = re.search(r"commissione[^.]{0,100}?(\d+)\s*(?:membri|componenti)", text, re.IGNORECASE)
    if m_comm:
        info_agg["commissione_giudicatrice"] = {
            "prevista": True,
            "numero_membri": int(m_comm.group(1)),
        }

    # Codice comportamento
    if "codice di comportamento" in text_lower:
        info_agg["codice_comportamento"] = True

    # Tracciabilità flussi
    if "tracciabilità" in text_lower and ("flussi" in text_lower or "136/2010" in text or "legge 136" in text_lower):
        info_agg["tracciabilita_flussi"] = True

    # ======================================================================
    # U) CCNL, FONTI FINANZIAMENTO, CONDIZIONI PNRR
    # ======================================================================

    # CCNL
    m_ccnl = re.search(r"(?:CCNL|C\.C\.N\.L\.?|contratto\s+collettivo\s+nazionale)[^\n]{0,200}", text, re.IGNORECASE)
    if m_ccnl:
        ccnl_text = _clean(m_ccnl.group(0))
        if ccnl_text and len(ccnl_text) > 10:
            info_agg["CCNL"] = ccnl_text[:300]

    # Fonti di finanziamento (raccogliamo tutte le fonti menzionate)
    fonti = []
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

    # Condizioni occupazionali PNRR / clausole sociali
    if "pnrr" in text_lower:
        pnrr = {}
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

    # Documentazione richiesta - conta documenti
    doc_section = _section_text(text, ["DOCUMENTAZIONE AMMINISTRATIVA", "BUSTA A", "14. CONTENUTO"], ["OFFERTA TECNICA", "BUSTA B", "15."], max_len=15000)
    if doc_section:
        # Conta items numerati
        doc_items = re.findall(r"(?:^|\n)\s*(\d{1,2})\s*[\).\-]\s+([^\n]{10,200})", doc_section)
        if doc_items:
            max_n = max(int(n) for n, _ in doc_items)
            doc["numero_documenti_richiesti"] = max_n

    # ======================================================================
    # PULIZIA FINALE: rimuovi sezioni vuote
    # ======================================================================

    def _clean_empty(d):
        if isinstance(d, dict):
            cleaned = {}
            for k, v in d.items():
                cv2 = _clean_empty(v)
                if cv2 is not None and cv2 != {} and cv2 != [] and cv2 != "":
                    cleaned[k] = cv2
            return cleaned if cleaned else None
        elif isinstance(d, list):
            cleaned = [_clean_empty(item) for item in d if _clean_empty(item) is not None]
            return cleaned if cleaned else None
        else:
            return d

    result = _clean_empty(result) or {}
    return result


# ---------------------------------------------------------------------------
# 5b. Flatten per pipeline / UI
# ---------------------------------------------------------------------------
def flatten_for_pipeline(nested: dict) -> tuple[dict, dict, dict]:
    """
    Converte il JSON nested profondo di extract_rules_based()
    nel formato piatto usato dalla pipeline e dalla UI.

    Returns:
        (flat_result, snippets, methods)
    """
    flat = {}
    methods = {}
    snippets = {}

    def _g(*keys):
        """Navigate nested dict with fallback."""
        obj = nested
        for k in keys:
            if isinstance(obj, dict):
                obj = obj.get(k)
            else:
                return None
        return obj

    # ── Identificativi ──
    ig = nested.get("informazioni_generali", {})
    sa = ig.get("stazione_appaltante", {}) if isinstance(ig.get("stazione_appaltante"), dict) else {}
    rup_d = ig.get("RUP", ig.get("rup", {}))

    flat["numero_bando"] = ig.get("numero_bando")
    flat["cig"] = ig.get("CIG")
    flat["cup"] = ig.get("CUP")
    flat["cui"] = ig.get("CUI")
    flat["cpv_principale"] = ig.get("CPV_principale")
    flat["cpv_secondari"] = ig.get("CPV_secondari", [])
    flat["codice_nuts"] = ig.get("codice_NUTS")

    det = ig.get("determina_a_contrarre", ig.get("determina", {}))
    if isinstance(det, dict):
        num = det.get("numero", "")
        data = det.get("data", "")
        flat["determina_a_contrarre"] = f"n. {num} del {data}" if num else None
    elif det:
        flat["determina_a_contrarre"] = str(det)
    else:
        flat["determina_a_contrarre"] = None

    # ── Stazione Appaltante e RUP ──
    flat["stazione_appaltante"] = sa.get("denominazione")
    flat["tipo_ente"] = sa.get("tipo_ente")
    flat["sa_indirizzo"] = sa.get("indirizzo")
    flat["sa_pec"] = sa.get("PEC") or sa.get("pec")
    flat["sa_email"] = sa.get("email")
    flat["sa_telefono"] = sa.get("telefono")
    flat["sa_sito_web"] = sa.get("sito_web")
    flat["sa_area_direzione"] = sa.get("area_direzione")
    flat["rup"] = rup_d.get("nome") if isinstance(rup_d, dict) else rup_d
    flat["rup_qualifica"] = rup_d.get("qualifica") if isinstance(rup_d, dict) else None
    flat["rup_email"] = rup_d.get("email") if isinstance(rup_d, dict) else None
    flat["rup_pec"] = rup_d.get("PEC") if isinstance(rup_d, dict) else None

    # ── Oggetto ──
    flat["oggetto_appalto"] = ig.get("titolo")
    flat["oggetto_sintetico"] = ig.get("oggetto_sintetico")

    # ── Tipo procedura ──
    tp = nested.get("tipo_procedura", {})
    flat["tipo_procedura"] = tp.get("tipo")
    flat["ambito_procedura"] = tp.get("ambito")
    flat["criterio_aggiudicazione"] = tp.get("criterio_aggiudicazione")
    flat["metodo_oepv"] = tp.get("metodo_OEPV")
    flat["rif_normativo_procedura"] = tp.get("riferimento_normativo")
    flat["inversione_procedimentale"] = tp.get("inversione_procedimentale", False)

    aq = tp.get("accordo_quadro", {})
    if isinstance(aq, dict) and aq.get("presente"):
        aq_parts = [aq.get("tipo", "")]
        if aq.get("durata_mesi"):
            aq_parts.append(f"{aq['durata_mesi']} mesi")
        flat["accordo_quadro"] = " - ".join(filter(None, aq_parts)) or "Si"
    elif isinstance(aq, bool) and aq:
        flat["accordo_quadro"] = "Si"
    else:
        flat["accordo_quadro"] = None

    flat["concessione"] = tp.get("concessione", False)

    # ── Piattaforma Telematica ──
    pt = nested.get("piattaforma_telematica", {})
    flat["piattaforma_nome"] = pt.get("nome")
    flat["piattaforma_url"] = pt.get("url")
    flat["piattaforma_gestore"] = pt.get("gestore")

    # ── Suddivisione Lotti ──
    sl = nested.get("suddivisione_lotti", {})
    flat["numero_lotti"] = sl.get("numero_lotti")
    flat["lotto_unico_motivazione"] = sl.get("lotto_unico_motivazione")
    flat["max_lotti_aggiudicabili"] = sl.get("numero_massimo_lotti_aggiudicabili")
    flat["vincoli_lotti"] = sl.get("vincoli_partecipazione_lotti")
    flat["lotti"] = sl.get("lotti", [])

    # ── Importi Complessivi ──
    ic = nested.get("importi_complessivi", {})
    flat["importo_totale_gara"] = _format_euro(ic.get("importo_totale_gara"))
    flat["importo_soggetto_ribasso"] = _format_euro(
        ic.get("importo_totale_soggetto_ribasso") or ic.get("importo_base_gara")
    )
    flat["importo_non_soggetto_ribasso"] = _format_euro(ic.get("importo_totale_non_soggetto_ribasso"))
    flat["importo_lavori"] = _format_euro(ic.get("importo_lavori_complessivo"))
    flat["oneri_sicurezza"] = _format_euro(ic.get("oneri_sicurezza"))
    flat["costi_manodopera"] = _format_euro(ic.get("costi_manodopera"))
    flat["quota_ribassabile_percentuale"] = ic.get("quota_ribassabile_percentuale")

    ant = ic.get("anticipazione", ic.get("anticipazione_prezzo", {}))
    if isinstance(ant, dict):
        if ant.get("prevista"):
            flat["anticipazione"] = f"{ant.get('percentuale', '')}%"
        else:
            flat["anticipazione"] = None
    elif ant:
        flat["anticipazione"] = _format_euro(ant) if isinstance(ant, (int, float)) else str(ant)
    else:
        flat["anticipazione"] = None

    rev = ic.get("revisione_prezzi", {})
    if isinstance(rev, dict) and rev.get("ammessa"):
        parts = ["Ammessa"]
        if rev.get("soglia_percentuale"):
            parts.append(f"soglia {rev['soglia_percentuale']}%")
        flat["revisione_prezzi"] = " - ".join(parts)
    elif isinstance(rev, dict) and rev.get("ammessa") is False:
        flat["revisione_prezzi"] = "Non ammessa"
    elif isinstance(rev, bool):
        flat["revisione_prezzi"] = "Ammessa" if rev else None
    else:
        flat["revisione_prezzi"] = None

    # ── Durata Contratto ──
    dur = nested.get("durata_contratto", {})
    if isinstance(dur, dict):
        if dur.get("durata_totale_anni") or dur.get("anni"):
            y = dur.get("durata_totale_anni") or dur.get("anni")
            flat["durata_contratto"] = f"{y} anni"
        elif dur.get("durata_totale_mesi") or dur.get("mesi"):
            m = dur.get("durata_totale_mesi") or dur.get("mesi")
            flat["durata_contratto"] = f"{m} mesi"
        elif dur.get("durata_totale_giorni") or dur.get("giorni"):
            g = dur.get("durata_totale_giorni") or dur.get("giorni")
            flat["durata_contratto"] = f"{g} giorni"
        else:
            flat["durata_contratto"] = None
        flat["decorrenza"] = dur.get("decorrenza")
        fp = dur.get("fase_progettazione_mesi") or dur.get("fase_realizzazione_mesi")
        flat["fase_progettazione"] = f"{fp} mesi" if fp else None
        fl = dur.get("fase_esecuzione_lavori_mesi")
        flat["fase_lavori"] = f"{fl} mesi" if fl else None
        fg_a = dur.get("fase_gestione_anni")
        fg_m = dur.get("fase_gestione_mesi")
        if fg_a:
            flat["fase_gestione"] = f"{fg_a} anni"
        elif fg_m:
            flat["fase_gestione"] = f"{fg_m} mesi"
        else:
            flat["fase_gestione"] = None
        pr = dur.get("proroga", {})
        if isinstance(pr, dict) and pr.get("ammessa"):
            flat["proroga"] = f"Ammessa - {pr.get('durata_mesi', '')} mesi" if pr.get("durata_mesi") else "Ammessa"
        else:
            flat["proroga"] = None
        sds = dur.get("societa_di_scopo", {})
        if isinstance(sds, dict) and sds.get("obbligatoria"):
            flat["societa_di_scopo"] = sds.get("forma") or "Obbligatoria"
        else:
            flat["societa_di_scopo"] = None
    else:
        flat["durata_contratto"] = str(dur) if dur else None
        flat["decorrenza"] = None
        flat["fase_progettazione"] = None
        flat["fase_lavori"] = None
        flat["fase_gestione"] = None
        flat["proroga"] = None
        flat["societa_di_scopo"] = None

    # ── Tempistiche ──
    temp = nested.get("tempistiche", {})
    flat["scadenza_offerte"] = temp.get("scadenza_offerte")
    flat["apertura_buste"] = temp.get("apertura_buste")
    flat["scadenza_chiarimenti"] = temp.get("termine_chiarimenti") or temp.get("scadenza_chiarimenti")
    flat["validita_offerta_giorni"] = temp.get("validita_offerta_giorni")
    flat["termine_stipula_giorni"] = temp.get("termine_stipula_giorni")
    # Standstill from tempistiche or aggiudicazione
    agg_raw = nested.get("aggiudicazione", {})
    stip_raw = agg_raw.get("stipula_contratto", {}) if isinstance(agg_raw, dict) else {}
    flat["standstill_giorni"] = (
        temp.get("standstill_giorni")
        or (stip_raw.get("termine_standstill_giorni") if isinstance(stip_raw, dict) else None)
    )

    # ── Requisiti Partecipazione ──
    rp = nested.get("requisiti_partecipazione", {})
    sogg = rp.get("soggetti_ammessi", {})
    flat["obbligo_giovane"] = sogg.get("obbligo_giovane_professionista", False) if isinstance(sogg, dict) else False

    # Figure professionali dal gruppo di lavoro
    gdl = rp.get("gruppo_di_lavoro", {})
    fig_list = gdl.get("figure_professionali", [])
    flat["numero_figure_richieste"] = len(fig_list) if fig_list else None
    flat["figure_professionali"] = [
        f.get("ruolo", "") + (f" ({f['requisiti']})" if f.get("requisiti") else "")
        for f in fig_list if isinstance(f, dict) and f.get("ruolo")
    ] if fig_list else []
    flat["ruoli_cumulabili"] = gdl.get("ruoli_cumulabili", False)

    cef = rp.get("capacita_economico_finanziaria", {})
    fg = cef.get("fatturato_globale", cef.get("fatturato_globale_minimo", {}))
    if isinstance(fg, dict):
        flat["fatturato_globale_minimo"] = _format_euro(fg.get("importo_minimo"))
        flat["fatturato_periodo"] = fg.get("periodo_riferimento")
    elif fg:
        flat["fatturato_globale_minimo"] = _format_euro(fg)
        flat["fatturato_periodo"] = None
    else:
        flat["fatturato_globale_minimo"] = None
        flat["fatturato_periodo"] = cef.get("periodo_riferimento_anni")

    fs = cef.get("fatturato_specifico", cef.get("fatturato_specifico_minimo", {}))
    flat["fatturato_specifico_minimo"] = _format_euro(fs.get("importo_minimo") if isinstance(fs, dict) else fs) if fs else None

    cop = cef.get("copertura_assicurativa", {})
    flat["copertura_assicurativa"] = _format_euro(cop.get("importo_minimo") if isinstance(cop, dict) else cop) if cop else None

    # ── Categorie e SOA ──
    ctp = rp.get("capacita_tecnico_professionale", {})

    soa = ctp.get("attestazione_SOA", ctp.get("requisiti_SOA", {}))
    if isinstance(soa, dict):
        flat["soa_richiesta"] = soa.get("richiesta", False)
        soa_cats = soa.get("categorie", [])
        flat["soa_categorie"] = [
            f"{c.get('id_categoria', c.get('categoria', ''))} - cl. {c.get('classifica', c.get('tipo', ''))}"
            for c in soa_cats if isinstance(c, dict)
        ] if soa_cats else []
    elif soa:
        flat["soa_richiesta"] = True
        flat["soa_categorie"] = [str(soa)] if not isinstance(soa, list) else soa
    else:
        flat["soa_richiesta"] = False
        flat["soa_categorie"] = []

    # Categorie opere con importi (formattate per la UI)
    raw_cats = ig.get("categorie_trovate", [])
    formatted_cats = []
    for c in raw_cats:
        if isinstance(c, dict):
            cat_id = c.get("id_categoria", c.get("categoria", ""))
            desc = c.get("descrizione", "")
            imp = c.get("importo_opera", c.get("importo", 0))
            imp_str = _format_euro(imp) if imp else ""
            parts = [cat_id]
            if desc:
                parts.append(desc[:80])
            if imp_str:
                parts.append(imp_str)
            formatted_cats.append(" - ".join(parts))
        elif isinstance(c, str):
            formatted_cats.append(c)
    flat["categorie_opere"] = formatted_cats

    # ── Criteri Valutazione ──
    cv = nested.get("criteri_valutazione", {})
    ot = cv.get("offerta_tecnica", {})
    oe = cv.get("offerta_economica", {})

    flat["punteggio_tecnica"] = ot.get("punteggio_massimo") or ot.get("punteggio_max")
    flat["punteggio_economica"] = oe.get("punteggio_massimo") or oe.get("punteggio_max")
    flat["soglia_sbarramento"] = ot.get("soglia_sbarramento")
    flat["riparametrazione"] = ot.get("riparametrazione", False)
    flat["formula_economica"] = oe.get("formula")
    flat["modalita_offerta"] = oe.get("modalita_offerta")
    flat["cifre_decimali"] = oe.get("cifre_decimali")

    criteri_list = ot.get("criteri", [])
    flat["criteri_tecnici"] = [
        {"codice": c.get("codice", ""), "nome": c.get("descrizione", c.get("nome", "")), "punteggio": c.get("punteggio", 0)}
        for c in criteri_list
    ] if criteri_list else []

    va = cv.get("verifica_anomalia", {})
    flat["verifica_anomalia"] = va.get("prevista") if isinstance(va, dict) else bool(va)

    # ── Offerta Tecnica Formato ──
    otf = nested.get("offerta_tecnica_formato", {})
    flat["ot_formato_pagina"] = otf.get("formato_pagina")
    flat["ot_carattere"] = otf.get("carattere")
    flat["ot_interlinea"] = otf.get("interlinea")
    flat["ot_limite_pagine"] = otf.get("pagine_massime") or otf.get("limite_pagine_totale")

    # ── Garanzie ──
    gar = nested.get("garanzie", {})
    gp = gar.get("garanzia_provvisoria", {})
    gd = gar.get("garanzia_definitiva", {})
    pol = gar.get("polizza_RC_professionale", gar.get("polizza_professionale", {}))

    flat["gar_provvisoria_dovuta"] = gp.get("dovuta") if isinstance(gp, dict) else None
    flat["gar_provvisoria_percentuale"] = f"{gp['percentuale']}%" if isinstance(gp, dict) and gp.get("percentuale") else None
    flat["gar_provvisoria_importo"] = _format_euro(gp.get("importo")) if isinstance(gp, dict) else None
    flat["gar_provvisoria_durata"] = gp.get("durata_giorni") if isinstance(gp, dict) else None
    flat["gar_definitiva_percentuale"] = f"{gd['percentuale']}%" if isinstance(gd, dict) and gd.get("percentuale") else None
    flat["gar_definitiva_forma"] = gd.get("forma") if isinstance(gd, dict) else None
    flat["polizza_rc"] = pol.get("richiesta", False) if isinstance(pol, dict) else bool(pol)
    flat["polizza_rc_copertura"] = pol.get("copertura") if isinstance(pol, dict) else None

    # ── Subappalto ──
    sub = nested.get("subappalto", {})
    flat["subappalto_ammesso"] = sub.get("ammesso") if isinstance(sub, dict) else None
    flat["subappalto_limite"] = f"{sub['limite_percentuale']}%" if isinstance(sub, dict) and sub.get("limite_percentuale") else None
    flat["subappalto_condizioni"] = sub.get("condizioni") if isinstance(sub, dict) else None

    # ── Avvalimento ──
    avv = nested.get("avvalimento", {})
    flat["avvalimento_ammesso"] = avv.get("ammesso") if isinstance(avv, dict) else None
    flat["avvalimento_condizioni"] = avv.get("condizioni") if isinstance(avv, dict) else None

    # ── Sopralluogo ──
    sop = nested.get("sopralluogo", {})
    flat["sopralluogo_obbligatorio"] = sop.get("obbligatorio", False) if isinstance(sop, dict) else bool(sop)
    flat["sopralluogo_modalita"] = sop.get("modalita") if isinstance(sop, dict) else None
    flat["sopralluogo_contatti"] = sop.get("contatti_prenotazione") or (sop.get("contatti") if isinstance(sop, dict) else None)
    flat["sopralluogo_termine"] = sop.get("termine") if isinstance(sop, dict) else None

    # ── Documentazione Amministrativa ──
    doc = nested.get("documentazione_amministrativa", {})
    dgue = doc.get("DGUE", {})
    flat["dgue_richiesto"] = dgue.get("richiesto", False) if isinstance(dgue, dict) else bool(dgue)
    anac = doc.get("contributo_ANAC", {})
    flat["contributo_anac"] = _format_euro(anac.get("importo_totale") or anac.get("importo") if isinstance(anac, dict) else anac)
    bollo = doc.get("imposta_bollo", {})
    flat["imposta_bollo"] = _format_euro(bollo.get("importo") if isinstance(bollo, dict) else bollo)
    flat["numero_documenti"] = doc.get("numero_documenti_richiesti")

    # ── Soccorso Istruttorio ──
    socc = nested.get("soccorso_istruttorio", {})
    flat["soccorso_ammesso"] = socc.get("ammesso", socc.get("previsto", False)) if isinstance(socc, dict) else bool(socc)
    flat["soccorso_termine_giorni"] = socc.get("termine_giorni", socc.get("termine")) if isinstance(socc, dict) else None
    flat["soccorso_riferimento"] = socc.get("riferimento") if isinstance(socc, dict) else None

    # ── Cause Esclusione ──
    ce = nested.get("cause_esclusione", {})
    auto_ce = ce.get("automatiche", {})
    flat["esclusione_automatiche"] = auto_ce.get("riferimento") if isinstance(auto_ce, dict) else (str(auto_ce) if auto_ce else None)
    nauto_ce = ce.get("non_automatiche", {})
    flat["esclusione_non_automatiche"] = nauto_ce.get("riferimento") if isinstance(nauto_ce, dict) else (str(nauto_ce) if nauto_ce else None)
    sc = ce.get("self_cleaning", {})
    flat["self_cleaning"] = sc.get("ammesso", False) if isinstance(sc, dict) else bool(sc)

    # ── Aggiudicazione e Stipula ──
    agg = nested.get("aggiudicazione", {})
    flat["max_lotti_concorrente"] = agg.get("numero_lotti_massimi_per_concorrente") or agg.get("max_lotti_per_concorrente")
    stip = agg.get("stipula_contratto", {})
    flat["stipula_forma"] = stip.get("forma") if isinstance(stip, dict) else None
    flat["stipula_standstill"] = stip.get("termine_standstill_giorni") if isinstance(stip, dict) else None
    flat["esecuzione_anticipata"] = stip.get("esecuzione_anticipata", False) if isinstance(stip, dict) else False

    # ── Penali ──
    pen = nested.get("penali", {})
    flat["penali_previste"] = pen.get("previste", False) if isinstance(pen, dict) else bool(pen)
    flat["penali_percentuale"] = f"{pen['percentuale_giornaliera']}%" if isinstance(pen, dict) and pen.get("percentuale_giornaliera") else None
    flat["penali_tetto"] = f"{pen['tetto_massimo_percentuale']}%" if isinstance(pen, dict) and pen.get("tetto_massimo_percentuale") else None
    flat["penali_descrizione"] = pen.get("descrizione") or pen.get("ritardo") if isinstance(pen, dict) else None

    # ── Sicurezza ──
    sic = nested.get("sicurezza", {})
    flat["sicurezza_oneri"] = _format_euro(sic.get("oneri_sicurezza_interferenza") or sic.get("oneri_sicurezza"))
    duvri = sic.get("DUVRI", {})
    flat["duvri_richiesto"] = duvri.get("richiesto") if isinstance(duvri, dict) else None
    flat["duvri_nota"] = duvri.get("nota") if isinstance(duvri, dict) else None

    # ── CAM Criteri Ambientali ──
    cam = nested.get("CAM_criteri_ambientali", {})
    flat["cam_applicabili"] = cam.get("applicabili", False) if isinstance(cam, dict) else bool(cam)
    flat["cam_decreto"] = cam.get("decreto_riferimento") if isinstance(cam, dict) else None
    flat["cam_requisiti"] = cam.get("requisiti_minimi") if isinstance(cam, dict) else None

    # ── Controversie ──
    cont = nested.get("controversie", {})
    flat["foro_competente"] = cont.get("foro_competente")
    flat["termine_ricorso_giorni"] = cont.get("termine_ricorso_giorni")
    flat["arbitrato"] = cont.get("arbitrato")
    cct = cont.get("collegio_consultivo_tecnico", {})
    flat["collegio_consultivo"] = cct.get("previsto", False) if isinstance(cct, dict) else bool(cct)

    # ── Tracciabilita e Info Aggiuntive ──
    trac = nested.get("tracciabilita_flussi", {})
    flat["tracciabilita_flussi"] = trac.get("obbligatoria", False) if isinstance(trac, dict) else bool(trac)

    ia = nested.get("informazioni_aggiuntive", {})
    flat["natura_servizio"] = ia.get("natura_servizio")
    flat["codice_comportamento"] = ia.get("codice_comportamento", False)
    flat["finanziamento"] = _g("informazioni_generali", "finanziamento", "fonte")
    flat["fonti_finanziamento"] = ia.get("fonti_finanziamento", [])
    flat["ccnl"] = ia.get("CCNL")

    comm = ia.get("commissione_giudicatrice", {})
    if isinstance(comm, dict) and comm.get("prevista"):
        flat["commissione_giudicatrice"] = f"Si - {comm.get('numero_membri', '?')} membri"
    else:
        flat["commissione_giudicatrice"] = None

    pnrr = ia.get("condizioni_PNRR", ia.get("condizioni_occupazionali_PNRR", {}))
    if isinstance(pnrr, dict):
        flat["pnrr_giovani"] = pnrr.get("quota_min_occupazione_giovanile_percentuale")
        flat["pnrr_donne"] = pnrr.get("quota_min_occupazione_femminile_percentuale")
        flat["dnsh"] = pnrr.get("DNSH", False)
    else:
        flat["pnrr_giovani"] = None
        flat["pnrr_donne"] = None
        flat["dnsh"] = False

    # ── Note operative (auto-generate) ──
    note = []
    if flat.get("sopralluogo_obbligatorio"):
        note.append("Sopralluogo obbligatorio")
    if flat.get("inversione_procedimentale"):
        note.append("Inversione procedimentale (art. 107 c.3)")
    if flat.get("soglia_sbarramento"):
        note.append(f"Soglia sbarramento tecnica: {flat['soglia_sbarramento']} punti")
    if flat.get("cam_applicabili"):
        note.append("Conformita CAM obbligatoria")
    if flat.get("finanziamento"):
        note.append(f"Finanziamento: {flat['finanziamento']}")
    if flat.get("soccorso_ammesso"):
        note.append("Soccorso istruttorio previsto")
    if flat.get("dnsh"):
        note.append("Principio DNSH (PNRR)")
    if flat.get("penali_previste"):
        note.append("Penali previste")
    if flat.get("subappalto_ammesso"):
        note.append("Subappalto ammesso")
    flat["note_operative"] = note

    # Build methods (tutte le chiavi con valore → "rules")
    for k, v in flat.items():
        if v is not None and v != "" and v != [] and v != {} and v is not False and v != 0:
            methods[k] = "rules"

    # Also keep the full nested result for reference
    flat["_nested_full"] = nested

    return flat, snippets, methods


def _format_euro(val) -> str | None:
    """Formatta un importo numerico come stringa Euro leggibile."""
    if val is None:
        return None
    try:
        num = float(val)
        if num <= 0:
            return None
        # Formato italiano: € 1.234.567,89
        formatted = f"{num:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"EUR {formatted}"
    except (ValueError, TypeError):
        return str(val) if val else None


# ---------------------------------------------------------------------------
# 5c. Estrazione da bytes (per integrazione pipeline)
# ---------------------------------------------------------------------------
def extract_from_pdf_bytes(pdf_bytes: bytes, filename: str = "upload.pdf") -> tuple[dict, dict, dict]:
    """
    Estrae dati da bytes PDF e ritorna nel formato pipeline.
    Usato dall'endpoint /api/extract.

    Returns:
        (flat_result, snippets, methods)
    """
    import tempfile, os
    # Scrivi bytes in file temporaneo
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        text = extract_text_from_pdf(tmp_path)
        nested = extract_rules_based(text)
        flat, snippets, methods = flatten_for_pipeline(nested)
        return flat, snippets, methods, text
    finally:
        os.unlink(tmp_path)


def extract_from_text_direct(text: str) -> tuple[dict, dict, dict]:
    """
    Estrae dati da testo e ritorna nel formato pipeline.

    Returns:
        (flat_result, snippets, methods)
    """
    nested = extract_rules_based(text)
    flat, snippets, methods = flatten_for_pipeline(nested)
    return flat, snippets, methods


# ---------------------------------------------------------------------------
# 6. Pipeline completa
# ---------------------------------------------------------------------------
def extract_disciplinare(
    pdf_path: str,
    provider: str = "rules",  # "openai", "anthropic", "rules"
    model: Optional[str] = None,
    save_output: bool = True,
) -> dict:
    """
    Pipeline completa di estrazione dati da un disciplinare.

    Args:
        pdf_path: Percorso del file PDF
        provider: "openai", "anthropic", "rules" (fallback senza LLM)
        model: Modello specifico (default: gpt-4o / claude-sonnet-4-20250514)
        save_output: Se salvare il JSON risultante accanto al PDF

    Returns:
        dict con tutti i dati estratti
    """
    print(f"[PDF] Estrazione da: {os.path.basename(pdf_path)}")

    # 1. Estrai testo
    print("  > Estrazione testo dal PDF...")
    pdf_text = extract_text_from_pdf(pdf_path)
    print(f"  > Estratti {len(pdf_text):,} caratteri")

    # 2. Estrai dati
    if provider == "rules":
        print("  > Estrazione con regole (senza LLM)...")
        result = extract_rules_based(pdf_text)
    else:
        print(f"  > Costruzione prompt per {provider}...")
        messages = build_extraction_prompt(pdf_text)

        print(f"  > Chiamata {provider} ({model or 'default'})...")
        if provider == "openai":
            raw = call_openai(messages, model=model or "gpt-4o")
        elif provider == "anthropic":
            raw = call_anthropic(messages, model=model or "claude-sonnet-4-20250514")
        else:
            raise ValueError(f"Provider non supportato: {provider}")

        # Parse JSON
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            # Prova ad estrarre il JSON dalla risposta
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {"errore": "Impossibile parsare la risposta", "raw": raw}

    # 3. Salva output
    if save_output:
        out_path = Path(pdf_path).with_suffix(".extracted.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  [OK] Salvato: {out_path.name}")

    return result


# ---------------------------------------------------------------------------
# 7. Batch processing
# ---------------------------------------------------------------------------
def extract_all_disciplinari(
    folder: str = ".",
    provider: str = "rules",
    model: Optional[str] = None,
) -> list[dict]:
    """Estrae dati da TUTTI i PDF disciplinari in una cartella."""
    pdf_files = sorted(Path(folder).glob("*.pdf"))
    # Filtra solo i disciplinari (escludi altri PDF)
    disciplinari = [
        p for p in pdf_files
        if any(kw in p.name.lower() for kw in ["disciplinare", "disciplinar", "bando"])
    ]

    if not disciplinari:
        disciplinari = pdf_files  # Fallback: usa tutti i PDF

    print(f"\n{'='*60}")
    print(f"  Estrazione batch: {len(disciplinari)} disciplinari")
    print(f"  Provider: {provider}")
    print(f"{'='*60}\n")

    results = []
    for i, pdf_path in enumerate(disciplinari, 1):
        print(f"\n[{i}/{len(disciplinari)}] {pdf_path.name}")
        print("-" * 50)
        try:
            result = extract_disciplinare(
                str(pdf_path), provider=provider, model=model
            )
            result["_source_file"] = pdf_path.name
            results.append(result)
        except Exception as e:
            print(f"  [ERR] ERRORE: {e}")
            results.append({"_source_file": pdf_path.name, "_error": str(e)})

    # Salva riepilogo
    summary_path = Path(folder) / "estrazione_riepilogo.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Riepilogo salvato: {summary_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Estrai dati strutturati da disciplinari di gara"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=".",
        help="Percorso PDF singolo o cartella con più PDF (default: cartella corrente)",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "rules"],
        default="rules",
        help="Provider LLM da usare (default: rules = estrazione senza LLM)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Modello LLM specifico (es. gpt-4o, claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Estrai da tutti i PDF nella cartella",
    )

    args = parser.parse_args()

    if args.all or os.path.isdir(args.input):
        folder = args.input if os.path.isdir(args.input) else "."
        extract_all_disciplinari(folder, provider=args.provider, model=args.model)
    else:
        extract_disciplinare(args.input, provider=args.provider, model=args.model)
