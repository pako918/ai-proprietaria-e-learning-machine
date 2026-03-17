"""Sezione A: Estrazione informazioni generali."""

import re
from .utils import _clean


def extract_info_generali(text: str, text_lower: str) -> dict:
    """Estrae informazioni generali: titolo, SA, RUP, CIG, CUP, ecc."""
    ig = {}

    # --- Titolo / Oggetto dell'appalto ---
    _titolo_text = text[:8000]
    titolo_patterns = [
        r"(?:OGGETTO\s+DELL[\u2019']?\s*APPALTO|OGGETTO\s+DELLA\s+GARA|OGGETTO\s+DELL[\u2019']?\s*AFFIDAMENTO|OGGETTO)\s*[:\s\u2013\-]+\s*([^\n]+(?:\n(?!CIG|CUP|Art\.|\d+[.)\s]+[A-Z]|\s*\n)[^\n]+)*)",
        r"(?:PROCEDURA\s+(?:APERTA|NEGOZIATA|RISTRETTA|COMPETITIVA)\s+(?:PER|RELATIVA\s+A)\s+(?:IL\s+|LA\s+|L[\u2019']|AL\s+|ALLA\s+)?)([^\n]+(?:\n(?!CIG|CUP|Pag\.|Disciplinare|DISCIPLINARE|Importo|\d+[.)\s]+[A-Z]|\s*\n)[^\n]+)*)",
        r"avente\s+(?:per|ad)\s+oggetto\s*[:\s]+([^\n]+(?:\n(?!CIG|CUP|Art\.|\s*\n)[^\n]+)*)",
        r"DISCIPLINARE\s+DI\s+GARA\s+(?:PER|RELATIV[OA]\s+A)\s+(?:IL\s+|LA\s+|L[\u2019'])?([^\n]+(?:\n(?!CIG|CUP|Pag\.|\s*\n)[^\n]+)*)",
        r"DISCIPLINARE\s+DI\s+GARA\s*\n\s*([^\n]{25,}(?:\n(?!\s*\(?CUP|\(?CIG|\d+[.)\s]+[A-Z]|\s*\n)[^\n]+)*)",
        r"(?:APPALTO|AFFIDAMENTO)\s+(?:DEI|DI|DEL|PER)\s+([^\n]+(?:\n(?!CIG|CUP|Pag\.|Importo|\s*\n)[^\n]+)*)",
        r"BANDO\s+DI\s+GARA\s+(?:PER|RELATIV[OA]\s+A)\s+(?:IL\s+|LA\s+|L[\u2019'])?([^\n]+(?:\n(?!CIG|CUP|Pag\.|\s*\n)[^\n]+)*)",
        r"GARA\s+(?:(?:EUROPEA|COMUNITARIA|PUBBLICA)\s+)?(?:PER|RELATIVA)\s+(?:A\s+)?(?:IL\s+|LA\s+|L[\u2019'])?([^\n]+(?:\n(?!CIG|CUP|Art\.|\s*\n)[^\n]+)*)",
        r"(?:^|\n)\s*OGGETTO\s*[:\n]\s*\n\s*([^\n]{10,500})",
        r"(?:\d+[.)\s]+)?(?:OGGETTO|Oggetto)\s+(?:dell[\u2019']?\s*)?(?:appalto|gara|affidamento|servizio|incarico)\s*[:.\-]*\n+\s*([^\n]{10,500})",
    ]
    _titolo_nope = {"importo", "euro ", "ribasso", "oneri sicurezza", "soggett",
                    "decreto", "prot.", "protocollo", "presidente dell", "c/o",
                    "segreteria@", "https://", "piattaforma", "telematic"}
    for pat in titolo_patterns:
        m = re.search(pat, _titolo_text, re.IGNORECASE)
        if m:
            titolo = _clean(m.group(1))
            if titolo and len(titolo) > 15:
                titolo = re.split(r"(?:Importo|CIG|CUP|Euro\s+[\d.]|Pag\.\s*\d)", titolo, flags=re.IGNORECASE)[0]
                titolo = _clean(titolo)
                if titolo and len(titolo) > 15 and not any(nw in titolo.lower() for nw in _titolo_nope):
                    ig["titolo"] = titolo[:500]
                    break

    if "titolo" not in ig:
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
            candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            ig["titolo"] = _clean(candidates[0][2])[:500]

    # --- Stazione appaltante ---
    sa = {}
    header = text[:5000]
    sa_patterns = [
        r"(?:STAZIONE\s+APPALTANTE|AMMINISTRAZIONE\s+AGGIUDICATRICE)\s*[:\n]\s*(.+?)(?:\n\n|\nAtti|\nPROCEDURA|\nDISCIPLINARE|\nCIG)",
        r"(?:L[''])(ATER\s+della\s+provincia\s+di\s+\w+)",
        r"(ATER\s+[-–]\s+Azienda\s+Territoriale[^\n]+)",
        r"(Comune\s+di\s+[A-Z][A-Za-z\s]+?)(?:\s*\(|\s*\n|\s*$|\s*,)",
        r"(Citt[àa]\s+Metropolitana\s+di\s+\w+(?:\s+di\s+\w+)?)",
        r"(Provincia\s+di\s+\w+)",
        r"(?:AMMINISTRAZIONE\s+DELEGANTE\s*\n?\s*)(COMUNE\s+DI\s+[A-Z]+(?:\s*\([A-Z]+\))?)",
        r"(Centrale\s+Unica\s+di\s+Committenza\s+[^\n]+)",
        r"(Azienda\s+(?:Sanitaria|Ospedaliera|ULSS|AUSL|USL|ASL)[^\n,;.]{3,80})",
    ]
    for pat in sa_patterns:
        m = re.search(pat, header, re.IGNORECASE | re.MULTILINE)
        if m:
            name = _clean(m.group(1))
            if name and len(name) > 3:
                sa["denominazione"] = name
                break

    # Fallback: search full text for University / public body name
    if not sa.get("denominazione"):
        # "Università degli Studi di X" with full name (partita IVA near it)
        m_univ = re.search(
            r"\b(Universit[àa]\s+degli\s+Studi\s+di\s+[A-Za-zÀ-ùÀ-ú\s]{3,60}?)"
            r"(?=\s*[-–]\s*(?:partita|p\.?\s*i\.?v|codice)|[,\s]*–|\s*\(|\n)",
            text, re.IGNORECASE,
        )
        if not m_univ:
            m_univ = re.search(
                r"\b(Universit[àa]\s+(?:degli\s+Studi\s+(?:di\s+)?|della\s+)[A-Za-zÀ-ùÀ-ú\s]{3,60}?)"
                r"(?=\s*–|\s*\(|\n|,)",
                text, re.IGNORECASE,
            )
        if m_univ:
            name = _clean(m_univ.group(1))
            if name and len(name) > 8:
                sa["denominazione"] = name

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
        elif "universit" in den_lower:
            sa["tipo_ente"] = "Università"
        elif "istituto" in den_lower or "agenzia" in den_lower:
            sa["tipo_ente"] = "Ente Pubblico"

    m_cuc = re.search(
        r"(?:CENTRALE\s+UNICA\s+DI\s+COMMITTENZA|C\.?U\.?C\.?)"
        r"(?:\s*[:\-\–]\s*|\s+(?:presso\s+(?:l['\u2019]\s*)?)?|\s+di\s+)"
        r"([^\n]{5,120})",
        header, re.IGNORECASE,
    )
    if m_cuc:
        cuc_name = _clean(m_cuc.group(1))
        # Evita di catturare solo acronimi (es. "CUC" ripetuto senza nome)
        # Evita di catturare riferimenti di protocollo o decreti
        _bad_cuc = ("prot.", "decreto", "n. 27", "n.27", "0020920", "https", "@", "pec")
        if (cuc_name and len(cuc_name) > 3
                and not re.fullmatch(r"C\.?U\.?C\.?", cuc_name, re.I)
                and not any(b in cuc_name.lower() for b in _bad_cuc)
                and not re.search(r"\d{5,}", cuc_name)):
            sa["CUC"] = cuc_name

    # Ente delegante (Comune che delega alla CUC)
    m_delegante = re.search(
        r"(?:AMMINISTRAZIONE\s+DELEGANTE|ENTE\s+DELEGANTE)[:\s\n–\-]+([^\n]{5,120})",
        header, re.IGNORECASE,
    )
    if m_delegante:
        delegante = _clean(m_delegante.group(1))
        if delegante and len(delegante) > 3:
            sa["ente_delegante"] = delegante

    m_pec = re.search(r"(?:PEC|pec)\s*[:\s]+([a-zA-Z0-9_.+-]+@(?:pec\.)[a-zA-Z0-9-]+\.[a-zA-Z.]+)", text)
    if m_pec:
        sa["PEC"] = m_pec.group(1).lower()

    m_email = re.search(r"(?:e-?mail|mail|Email)\s*[:\s]+([a-zA-Z0-9_.+-]+@(?!pec\.)[a-zA-Z0-9-]+\.[a-zA-Z.]+)", text, re.IGNORECASE)
    if m_email:
        sa["email"] = m_email.group(1).lower()

    m_web = re.search(r"(?:sito|website|web)\s*[:\s]*(https?://[^\s\n]+)", text, re.IGNORECASE)
    if m_web:
        sa["sito_web"] = m_web.group(1)

    m_dir = re.search(r"(?:DIREZIONE|SETTORE|AREA|SERVIZIO)\s+([A-Z][^\n]{5,60})", header)
    if m_dir:
        dir_text = _clean(m_dir.group(0))
        if dir_text and ig.get("titolo") and dir_text[:30].lower() not in ig["titolo"][:100].lower():
            sa["area_direzione"] = dir_text

    # Sede SA / ente delegante
    _sede_patterns = [
        r"(?:sede[^::\n]{0,20}[:\-]\s*)([Vv]ia\s+[^\n,;]{5,80}[,\s]+\d{5}[^\n,;]{0,60})",
        r"([Vv]ia\s+[A-Z][^\n,;]{3,60}[,\s]+\d{5}(?:[^\n,;]{0,40})?)",
        r"([Pp]iazza\s+[A-Z][^\n,;]{3,60}[,\s]+\d{5}(?:[^\n,;]{0,40})?)",
    ]
    _sede_nope = ("https", "pec", "@", "decreto", "prot.")
    for _sp in _sede_patterns:
        _m_sede = re.search(_sp, header, re.IGNORECASE)
        if _m_sede:
            _sv = _clean(_m_sede.group(1))
            if _sv and len(_sv) > 8 and not any(b in _sv.lower() for b in _sede_nope):
                sa["sede"] = _sv[:200]
                break

    if sa:
        ig["stazione_appaltante"] = sa

    # --- RUP (singolo o doppio: programmazione+esecuzione vs affidamento CUC) ---
    _rup_false = {"responsabile", "procedimento", "progetto", "documento",
                  "informatico", "firmato", "digitale", "unico", "servizio",
                  "direzione", "settore", "procedura",
                  "email", "pec", "tel", "telefono", "fax", "indirizzo"}
    _rup_scan_patterns = [
        r"(?:R\.?U\.?P\.?)\s*[:\s]+(?:[Rr]esponsabile[^\n]*?,\s*)?(?:(?:Dott\.?(?:ssa)?|Ing\.?|Arch\.?|Geom\.?|Prof\.?)\s+)?([A-Z][a-zàèéìòù]+(?:[ \t]+[A-Z][a-zàèéìòù]+){1,3})",
        r"[Rr]esponsabile\s+(?:[Uu]nico\s+)?(?:del\s+)?(?:[Pp]rogetto|[Pp]rocedimento|procedimento\s+e\s+del\s+progetto)[,.\s:]+(?:(?:(?:è|e)\s+)?(?:il\s+|la\s+)?)?(?:(?:Dott\.?(?:ssa)?|Ing\.?|Arch\.?|Geom\.?|Prof\.?)\s+)?([A-Z][a-zàèéìòù]+(?:[ \t]+[A-Z][a-zàèéìòù]+){1,3})",
        r"(?:R\.?U\.?P\.?|Responsabile\s+(?:Unico\s+)?(?:del\s+)?(?:Progetto|Procedimento))[\s\S]{0,400}?(?:[Dd]ott\.?(?:ssa)?|[Ii]ng\.?|[Aa]rch\.?|[Gg]eom\.?|[Pp]rof\.?)\s+([A-Z][a-zàèéìòù]+(?:[ \t]+[A-Z][a-zàèéìòù]+){1,3})",
        r"(?:R\.?U\.?P\.?|Responsabile\s+(?:Unico\s+)?(?:del\s+)?(?:Progetto|Procedimento))[\s\S]{0,400}?(?:(?:è|e)\s+)?l['\u2019]([Aa]rch|[Ii]ng|[Dd]ott)\.?\s+([A-Z][a-zàèéìòù]+(?:[ \t]+[A-Z][a-zàèéìòù]+){1,3})",
    ]
    _N_RUP_PATS = len(_rup_scan_patterns)

    def _rup_candidate(m, pi: int) -> str | None:
        raw = _clean(m.group(2) if pi == _N_RUP_PATS - 1 else m.group(1))
        if raw:
            raw = re.sub(r'\s+(?:Email|Pec|Tel|Telefono|Fax|Indirizzo|Documento|Firmato|Responsabile)\b.*', '', raw, flags=re.I)
            raw = re.sub(r'\s+(?:Documento|informatico|sottoscritt|digitale|firmato)\b.*', '', raw, flags=re.I)
            raw = _clean(raw)
        if raw and not any(w in raw.lower() for w in _rup_false):
            return raw
        return None

    def _build_rup(name: str, match_start: int) -> dict:
        area = text[max(0, match_start - 300): match_start + 300]
        r: dict = {"nome": name}
        for qual in ["Dott.ssa", "Dott.", "Ing.", "Arch.", "Geom.", "Prof."]:
            if qual.lower() in area.lower():
                r["qualifica"] = qual.rstrip(".")
                break
        m_mail = re.search(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z.]+)", area)
        if m_mail:
            r["email"] = m_mail.group(1).lower()
        return r

    rup_main: dict = {}
    rup_aff: dict = {}
    _seen_rup_names: set = set()

    for pi, pat in enumerate(_rup_scan_patterns):
        for m in re.finditer(pat, text, re.DOTALL):
            name = _rup_candidate(m, pi)
            if not name or name in _seen_rup_names:
                continue
            _seen_rup_names.add(name)
            ctx = text[max(0, m.start() - 300): m.end() + 300].lower()
            is_affid = bool(re.search(
                r"affidamento|centrale\s+unica|c\.?u\.?c\b|montedoro|unione\s+dei\s+comuni|stazione\s+appaltante\s+delegat",
                ctx,
            ))
            is_comune = bool(re.search(
                r"comune\s+di\s+turi|ente\s+delegante|programmazione|progettazione[^\w].*esecuzione",
                ctx,
            ))
            rd = _build_rup(name, m.start())
            if is_affid and not is_comune and not rup_aff:
                rd["ruolo"] = "Affidamento (CUC)"
                rup_aff = rd
            elif (is_comune or not is_affid) and not rup_main:
                rd["ruolo"] = "Programmazione / Progettazione / Esecuzione"
                rup_main = rd
            elif is_affid and not rup_aff:
                rd["ruolo"] = "Affidamento (CUC)"
                rup_aff = rd
        if rup_main and rup_aff:
            break

    if rup_main:
        ig["RUP"] = rup_main
    if rup_aff:
        ig["RUP_CUC"] = rup_aff

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
    cigs = re.findall(r"CIG[:\s]*([A-Za-z0-9]{10,})", text)
    if not cigs:
        cig_spaced = re.findall(r"CIG[:\s]*((?:[A-Za-z0-9]{2,5}\s*){3,})", text)
        for cs in cig_spaced:
            cleaned = re.sub(r'\s+', '', cs)
            if len(cleaned) >= 10:
                cigs.append(cleaned)
    if not cigs:
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

    return ig
