"""
Sezione G – Criteri di valutazione.

Estrae punteggi offerta tecnica/economica, soglia sbarramento,
sub-criteri (codice/nome/punti), D/T, verifica anomalia.
"""

from __future__ import annotations
import re
from .utils import _clean, _section_text


def _pick_desc_after_code(cells: list[str], code: str) -> str | None:
    """Return the best desc cell that appears AFTER *code* in the row."""
    try:
        code_idx = cells.index(code)
    except ValueError:
        return None
    for c in cells[code_idx + 1:]:
        if len(c) > 5 and not re.match(r"^[\d,.]+$", c) and not re.match(r"^[A-Z](?:\.\d{1,2})?$", c):
            return c
    for c in cells:
        if c != code and len(c) > 8 and not re.match(r"^[\d,.]+$", c) and not re.match(r"^[A-Z](?:\.\d{1,2})?$", c):
            return c
    return None


def extract_valutazione(text: str, text_lower: str) -> dict:
    """
    Restituisce il dict ``criteri_valutazione`` con sotto-chiavi:

    * offerta_tecnica  (punteggio_massimo, soglia_sbarramento, riparametrazione, criteri)
    * offerta_economica (punteggio_massimo, modalita_offerta, cifre_decimali, formula)
    * verifica_anomalia
    """

    cv: dict = {"offerta_tecnica": {"criteri": []}, "offerta_economica": {}}

    crit_section = _section_text(
        text,
        ["CRITERIO DI AGGIUDICAZIONE", "CRITERI DI VALUTAZIONE", "18. CRITERIO", "5.1 Criteri"],
        ["SVOLGIMENTO DELLE OPERAZIONI DI GARA", "23.", "24.", "25.", "19."],
        max_len=20000,
    )

    ot = cv["offerta_tecnica"]
    oe = cv["offerta_economica"]

    # ── Punteggio offerta tecnica ────────────────────────────────────────
    pt_patterns = [
        r"offerta\s+tecnica\s*(?:[:=]|\(|punti\s+)?\s*(?:max\.?\s*|massimo\s+)?(?:punti\s+)?(\d{1,3})\s*(?:punti|pt|punto|/100|\))",
        r"(?:Punteggio\s+)?[Oo]fferta\s+tecnica[^\d]{0,30}(\d{1,3})\s*(?:punti|pt|/)",
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

    # ── Punteggio offerta economica ──────────────────────────────────────
    pe_patterns = [
        r"offerta\s+economica\s*(?:[:=]|\(|punti\s+)?\s*(?:max\.?\s*|massimo\s+)?(?:punti\s+)?(\d{1,3})\s*(?:punti|pt|punto|/100|\))",
        r"(?:Punteggio\s+)?[Oo]fferta\s+economica[^\d]{0,30}(\d{1,3})\s*(?:punti|pt|/)",
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

    # ── Soglia sbarramento ───────────────────────────────────────────────
    soglia_patterns = [
        r"soglia\s+(?:di\s+)?sbarramento[^0-9]{0,50}?(\d{1,3})",
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

    # ── Sub-criteri valutazione ──────────────────────────────────────────
    sub_criteri = re.findall(
        r"(?:^|\n)\s*([A-Z](?:\.\d{1,2})?|\d{1,2}(?:\.\d{1,2})?)\s*[.\-\)\s]\s*"
        r"(.{10,200}?)\s+"
        r"(\d{1,3}(?:[.,]\d{1,2})?)\s*(?:punti|pt|punto|\s+\d|\n|$)",
        crit_section,
        re.IGNORECASE | re.MULTILINE,
    )

    compact_criteri = re.findall(
        r"(?:^|\n)\s*([A-Z](?:\.\d{1,2})?)\s+(\d{1,3})\s*$",
        crit_section,
        re.MULTILINE,
    )
    existing_codes = {s[0] for s in sub_criteri}
    for code, pts in compact_criteri:
        if code not in existing_codes:
            sub_criteri.append((code, code, pts))

    # Criteri da TABELLA pdfplumber
    table_sections = re.findall(
        r"\[TABELLA[^\]]*\]\n(.*?)(?=\n\n|\n---|\n\[TABELLA|\Z)", crit_section, re.DOTALL,
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

    # ── Arricchimento descrizioni ────────────────────────────────────────
    _table_descs: dict[str, str] = {}
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

    for m_bold in re.finditer(
        r"\*\*\s*([A-Z](?:\.\d{1,2})?)\s*[\u2013\u2014\-–—]\s*(.{10,120}?)\s*\*\*",
        crit_section,
    ):
        code_b = m_bold.group(1)
        desc_b = m_bold.group(2).strip()
        if code_b not in _table_descs:
            _table_descs[code_b] = desc_b

    _known_names = {
        "A": "Qualità e adeguatezza del gruppo di lavoro proposto",
        "A.1": "Competenze dei tecnici obbligatori",
        "A.2": "Integrazione e potenziamento del team",
        "B": "Promozione dell'inserimento di giovani professionisti",
        "C": "Approccio metodologico, organizzativo e coordinamento tecnico",
        "D": "Tecniche di rilievo e restituzione grafica",
        "E": "Certificazioni",
        "E.1": "Certificazione ISO 9001 per il sistema di gestione della qualità",
        "E.2": "Certificazione di parità di genere ai sensi dell'art. 46-bis del D.Lgs. 198/2006",
    }

    enriched = []
    for codice, desc, punti_str in sub_criteri:
        table_desc = _table_descs.get(codice, "")
        is_sub = "." in codice
        if is_sub and table_desc and len(table_desc) >= 10:
            better_desc = table_desc
        elif table_desc and len(table_desc) > len(desc):
            better_desc = table_desc
        else:
            better_desc = desc
        if len(better_desc) < 10:
            m_name = re.search(
                rf"(?:^|\n)\s*{re.escape(codice)}\s+([A-Z][^\d\n]{{8,120}})",
                crit_section, re.MULTILINE,
            )
            if m_name:
                better_desc = m_name.group(1).strip()
        if codice in _known_names:
            better_desc = _known_names[codice]
        if not better_desc or len(better_desc) < 3:
            better_desc = codice
        enriched.append((codice, better_desc, punti_str))
    sub_criteri = enriched

    criteri_parsed: list[dict] = []
    for codice, desc, punti_str in sub_criteri:
        try:
            punti = float(punti_str.replace(",", "."))
        except ValueError:
            continue
        if punti <= 0 or punti > 100:
            continue
        desc_clean = _clean(desc)
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

    # Deduplica per codice
    _seen_codes: dict[str, dict] = {}
    for c in criteri_parsed:
        code = c["codice"]
        if code not in _seen_codes or c["punteggio"] > _seen_codes[code]["punteggio"]:
            _seen_codes[code] = c
    criteri_parsed = list(_seen_codes.values())

    criteri_filtered = [
        c for c in criteri_parsed
        if not re.search(r'(CRITERI DI VALUTAZIONE|METODO DI|CALCOLO DEI|PROCEDURA|OFFERTA TECNICA\s*\.)', c['nome'], re.IGNORECASE)
        and '...' not in c['nome']
        and len(c['nome']) > 8
    ]

    # ── Arricchimento: descrizione dettagliata da sez. 20.1 ─────────────
    _known_codes = {c["codice"] for c in criteri_filtered}
    _desc_rel: dict[str, str] = {}
    _all_rel = list(re.finditer(r'20\.1\s+RELAZIONE\s+TECNIC', text, re.IGNORECASE))
    m_start_rel = _all_rel[-1] if _all_rel else None
    if m_start_rel:
        _r0 = m_start_rel.end()
        m_end_rel = re.search(r'\n\s*(?:20\.2|21\.)\s+\w', text[_r0:], re.IGNORECASE)
        _r1 = _r0 + (m_end_rel.start() if m_end_rel else 5000)
        rel_body = text[_r0:_r1]
        _hdr_patt = re.compile(
            r'(?:^|\n)\s*([A-Z](?:\.\d{1,2})?)\s*[.–\-—\)]+\s+([A-Z\u00C0-\u017F][^\n]{5,})',
        )
        _hdrs = list(_hdr_patt.finditer(rel_body))
        for idx_h, hdr in enumerate(_hdrs):
            code_h = hdr.group(1)
            if code_h not in _known_codes:
                continue
            hdr_name = hdr.group(2).strip()
            h_end = _hdrs[idx_h + 1].start() if idx_h + 1 < len(_hdrs) else len(rel_body)
            chunk = rel_body[hdr.end():h_end].strip()
            if '.' in code_h and hdr_name:
                chunk = hdr_name + "\n" + chunk
            chunk = re.sub(r'---\s*Pagina\s+\d+\s*---', '', chunk)
            chunk = re.sub(r'\n\s*\d{1,3}\s*(?=\n)', '', chunk)
            chunk = re.sub(r'\n\d{1,3}\s*$', '', chunk.rstrip())
            chunk = re.sub(r'\s+\d{1,3}\s+(?=\d\.\s)', ' ', chunk)
            if chunk:
                _desc_rel[code_h] = _clean(chunk)[:2000]

    # ── Arricchimento: punteggi D/T dalla TABELLA ────────────────────────
    _dt_map: dict[str, dict] = {}
    _score_line_re = re.compile(
        r'\|\s*(\d*)\s*\|\s*\|\s*\|\s*(\d*)\s*\|\s*\|\s*\|\s*(\d*)\s*\|\s*\|?\s*$'
    )
    for tab in table_sections:
        _seen: list[str] = []
        for line in tab.split("\n"):
            if "|" not in line:
                continue
            cells = [c.strip() for c in line.split("|")]
            for cell in cells:
                if cell in _known_codes and cell not in _seen:
                    _seen.append(cell)
            ms = _score_line_re.search(line)
            if ms:
                total = int(ms.group(1)) if ms.group(1) else None
                d_val = int(ms.group(2)) if ms.group(2) else None
                t_val = int(ms.group(3)) if ms.group(3) else None
                mains = [c for c in _seen if '.' not in c]
                subs = [c for c in _seen if '.' in c]
                if subs:
                    sc = subs[-1]
                    _dt_map[sc] = {'d': d_val, 't': t_val}
                if mains and not subs:
                    _dt_map[mains[-1]] = {'d': d_val, 't': t_val}
                _seen = []

    # Fallback D/T nel testo non-tabella
    for c in criteri_filtered:
        code = c["codice"]
        if code not in _dt_map and '.' not in code:
            m_inline = re.search(
                rf'(?:^|\n)\s*{re.escape(code)}\s+[^\d\n]{{5,120}}\s+'
                rf'(\d{{1,3}})\s+(\d{{1,3}})\s*(?:\n|$)',
                crit_section, re.MULTILINE,
            )
            if m_inline:
                pts = int(m_inline.group(1))
                d_or_t = int(m_inline.group(2))
                if d_or_t == pts:
                    _dt_map[code] = {'d': d_or_t, 't': None}
                else:
                    _dt_map[code] = {'d': d_or_t, 't': pts - d_or_t if pts > d_or_t else None}

    # Applica arricchimento
    for c in criteri_filtered:
        code = c["codice"]
        if code in _desc_rel:
            c["descrizione_dettagliata"] = _desc_rel[code]
        dt = _dt_map.get(code, {})
        if '.' not in code:
            sum_d = sum(
                _dt_map.get(s["codice"], {}).get("d") or 0
                for s in criteri_filtered if s["codice"].startswith(code + ".")
            )
            sum_t = sum(
                _dt_map.get(s["codice"], {}).get("t") or 0
                for s in criteri_filtered if s["codice"].startswith(code + ".")
            )
            has_subs = any(s["codice"].startswith(code + ".") for s in criteri_filtered)
            if has_subs:
                if sum_d:
                    c["punteggio_discrezionale"] = float(sum_d)
                if sum_t:
                    c["punteggio_tabellare"] = float(sum_t)
            else:
                if dt.get("d"):
                    c["punteggio_discrezionale"] = float(dt["d"])
                if dt.get("t"):
                    c["punteggio_tabellare"] = float(dt["t"])
        else:
            if dt.get("d"):
                c["punteggio_discrezionale"] = float(dt["d"])
            if dt.get("t"):
                c["punteggio_tabellare"] = float(dt["t"])
        has_d = c.get("punteggio_discrezionale") is not None
        has_t = c.get("punteggio_tabellare") is not None
        if has_d and has_t:
            c["tipo"] = "misto"
        elif has_d:
            c["tipo"] = "discrezionale"
        elif has_t:
            c["tipo"] = "tabellare"

    if criteri_filtered:
        ot["criteri"] = criteri_filtered

    # ── Modalità offerta economica ───────────────────────────────────────
    if "ribasso percentuale" in text_lower:
        oe["modalita_offerta"] = "ribasso_percentuale"
    elif "prezzo" in text_lower and "offerta economica" in text_lower:
        oe["modalita_offerta"] = "prezzo"

    m_dec = re.search(r"(\d+)\s*cifre\s+decimali", text, re.IGNORECASE)
    if m_dec:
        oe["cifre_decimali"] = int(m_dec.group(1))

    m_form = re.search(r"(?:formula|metodo)[^.]{0,30}?(?:bilineare|non lineare|lineare|proporzionale)", text, re.IGNORECASE)
    if m_form:
        oe["formula"] = _clean(m_form.group(0))

    # ── Verifica anomalia ────────────────────────────────────────────────
    va: dict = {}
    if "anomal" in text_lower or "anormalmente bass" in text_lower:
        va["prevista"] = True
        m_va = re.search(r"(?:art\.?\s*110|art\.?\s*97)", text, re.IGNORECASE)
        if m_va:
            va["riferimento_normativo"] = _clean(m_va.group(0))
    if va:
        cv["verifica_anomalia"] = va

    return cv
