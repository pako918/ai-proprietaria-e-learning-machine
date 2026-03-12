"""
Sezione F – Requisiti di Partecipazione.

Estrae soggetti ammessi, fatturato, servizi analoghi, categorie,
gruppo di lavoro (figure professionali) e cumulabilità ruoli.
"""

from __future__ import annotations
import re
from .utils import _clean, _parse_euro, _section_text


# ── helpers interni ──────────────────────────────────────────────────────
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


# ── entry-point ──────────────────────────────────────────────────────────
def extract_requisiti(text: str, text_lower: str) -> dict:
    """
    Restituisce il dict ``requisiti_partecipazione`` con le sotto-chiavi:

    * soggetti_ammessi
    * capacita_economico_finanziaria
    * capacita_tecnico_professionale
    * gruppo_di_lavoro
    """

    rp: dict = {
        "soggetti_ammessi": {},
        "idoneita_professionale": {},
        "capacita_economico_finanziaria": {},
        "capacita_tecnico_professionale": {},
        "gruppo_di_lavoro": {"figure_professionali": []},
    }

    req_section = _section_text(
        text,
        ["REQUISITI DI ORDINE SPECIALE", "REQUISITI DI ORDINE GENERALE",
         "REQUISITI DI PARTECIPAZIONE", "6. REQUISITI"],
        ["7. AVVALIMENTO", "8. SUBAPPALTO", "AVVALIMENTO", "SUBAPPALTO", "GARANZI"],
        max_len=25000,
    )

    # ── Soggetti ammessi ─────────────────────────────────────────────────
    sogg = rp["soggetti_ammessi"]
    sogg_types: list[str] = []
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

    # ── Fatturato globale ────────────────────────────────────────────────
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

    m_fat_per = re.search(
        r"fatturato\s+globale[^.]{0,200}?((?:ultimo|miglio)\w*\s+\w+\s+(?:anni?|esercizi?)[^.]{0,50})",
        text, re.IGNORECASE,
    )
    if m_fat_per:
        if "fatturato_globale" not in fat_glob:
            fat_glob["fatturato_globale"] = {"richiesto": True}
        fat_glob["fatturato_globale"]["periodo_riferimento"] = _clean(m_fat_per.group(1))

    # ── Servizi analoghi / servizi di ingegneria ─────────────────────────
    srv = rp["capacita_tecnico_professionale"]
    m_srv2 = re.search(
        r"(?:Aver\s+)?regolarmente\s+eseguito[,\s]+nei\s+((?:dieci|cinque|\d+)\s+anni\s+antecedent[^,]{0,80})",
        text, re.IGNORECASE,
    )
    if m_srv2:
        srv["servizi_analoghi"] = {
            "richiesti": True,
            "periodo_riferimento": _clean(m_srv2.group(1)),
        }

    if not srv.get("servizi_analoghi"):
        m_srv = re.search(
            r"(?:servizi\s+(?:analoghi|di\s+punta)|"
            r"n\.\s*\d+\s*(?:\([^)]*\)\s*)?servizi\s+di\s+ingegneria\s+e\s+(?:di\s+)?architettura)"
            r"[^.]{0,200}?((?:ultimo|ultimi|dieci|cinque)\s+\w+\s*\w*(?:\s+anni?\s*\w*)?)",
            text, re.IGNORECASE,
        )
        if m_srv:
            srv["servizi_analoghi"] = {
                "richiesti": True,
                "periodo_riferimento": _clean(m_srv.group(1)),
            }

    # ── Requisiti categorie per servizi analoghi ─────────────────────────
    cat_req = re.findall(
        r"(?:categori\w+|class\w+)\s+(E\.?\d{2}|S\.?\d{2}|IA\.?\d{2}|D\.?\d{2}|V\.?\d{2})[^.]{0,100}?"
        r"(?:import\w+[^€\d]{0,30}[€\s]*([\d.,]+)|class\w+\s+(\w+))",
        req_section, re.IGNORECASE,
    )
    if cat_req:
        cats_list: list[dict] = []
        for cat_id, imp_val, classe in cat_req:
            entry: dict = {"categoria": cat_id[:1] + "." + cat_id[-2:] if "." not in cat_id else cat_id}
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

    # ── Fallback: tabella pipe-delimited (CATEGORIA D'OPERA) ─────────────
    if not srv.get("servizi_analoghi", {}).get("categorie_richieste"):
        _tab_cats_primary: list[dict] = []
        _tab_cats_fallback: list[dict] = []
        for tab_m in re.finditer(
            r"\[TABELLA[^\]]*\]\n(.*?)(?=\n\n|\n---|\n\[TABELLA|\Z)", text, re.DOTALL,
        ):
            tab_text = tab_m.group(1)
            if "CATEGORIA" not in tab_text.upper():
                continue
            is_servizi_punta = "Importo complessivo" in tab_text or "lavori progettati" in tab_text
            target_list = _tab_cats_primary if is_servizi_punta else _tab_cats_fallback
            lines = tab_text.split("\n")
            acc_desc = ""
            in_header = True
            for line in lines:
                if "|" in line:
                    parts = [p.strip() for p in line.split("|")]
                    codes_in_row = [p for p in parts if re.match(r"^(?:E|S|IA|D|V)\.?\d{2}$", p)]
                    if codes_in_row:
                        in_header = False
                        imp_cell = None
                        grado_cell = None
                        desc_cell = acc_desc
                        for p in parts:
                            if p in codes_in_row or not p:
                                continue
                            if re.match(r"^€?\s*[\d.,]+(?:\s*€)?$", p) and len(p) > 4:
                                v = _parse_euro(p)
                                if v and v > 10000:
                                    imp_cell = v
                                elif v and v < 10:
                                    grado_cell = v
                            elif re.match(r"^\d+[.,]\d{1,2}$", p):
                                grado_cell = float(p.replace(",", "."))
                            elif len(p) > 10 and not re.match(r"^[\d.,€\s]+$", p):
                                desc_cell = (desc_cell + " " + p).strip() if desc_cell else p
                        for code in codes_in_row:
                            code_norm = code if "." in code else code[:len(code)-2] + "." + code[-2:]
                            entry = {"categoria": code_norm}
                            if desc_cell:
                                entry["descrizione"] = _clean(desc_cell)[:200]
                            if imp_cell:
                                entry["importo_minimo"] = imp_cell
                            if grado_cell:
                                entry["grado_complessita"] = grado_cell
                            if not any(c.get("categoria") == code_norm for c in target_list):
                                target_list.append(entry)
                        acc_desc = ""
                    elif not in_header:
                        desc_parts = [p for p in parts if p and len(p) > 5
                                      and not re.match(r"^[\d.,€\s]+$", p)
                                      and "TOTALE" not in p.upper()]
                        if desc_parts:
                            acc_desc = (acc_desc + " " + " ".join(desc_parts)).strip()
                else:
                    stripped = line.strip()
                    if in_header and re.match(
                        r"(?:EDILIZIA|IMPIANTI|STRUTTUR|IDRAULIC|VIABILIT|INFRASTRUTTUR)",
                        stripped, re.IGNORECASE,
                    ):
                        in_header = False
                    if not in_header and stripped and len(stripped) > 3 and "TOTALE" not in stripped.upper():
                        acc_desc = (acc_desc + " " + stripped).strip() if acc_desc else stripped

        chosen = _tab_cats_primary or _tab_cats_fallback
        if chosen:
            if "servizi_analoghi" not in srv:
                srv["servizi_analoghi"] = {"richiesti": True}
            srv["servizi_analoghi"]["categorie_richieste"] = chosen

    # ── Fallback 2: bold-line format (non-pipe tables) ───────────────────
    if not srv.get("servizi_analoghi", {}).get("categorie_richieste"):
        _bold_cats: list[dict] = []
        bold_section_match = re.search(
            r"(?:Importo\s+complessivo\s+dei\s+lavori\s+progettati|"
            r"max\s+di\s+n\.\s*\d+\s+servizi|"
            r"REQUISITI\s+DI\s+CAPACIT[^\n]*TECNICA)",
            text, re.IGNORECASE,
        )
        if bold_section_match:
            scan_start = bold_section_match.start()
            scan_text = text[scan_start:scan_start + 5000]
            acc_desc = ""
            scan_lines = scan_text.split("\n")
            for line_idx, line in enumerate(scan_lines):
                stripped = line.strip()
                code_m = re.match(r'^[*\s]*((?:E|S|IA|D|V)\.?\d{2})\s*\**\s*$', stripped)
                if code_m:
                    code_raw = code_m.group(1)
                    code_norm = code_raw if "." in code_raw else code_raw[:-2] + "." + code_raw[-2:]
                    imp_val = None
                    grado_val = None
                    for la_line in scan_lines[line_idx + 1:line_idx + 5]:
                        la_clean = la_line.strip().replace("**", "").strip()
                        if not la_clean:
                            continue
                        if "TOTALE" in la_clean.upper():
                            break
                        v = _parse_euro(la_clean.replace("Ç", "€"))
                        if v and v > 100:
                            imp_val = v
                            break
                        elif v and 0 < v < 10:
                            grado_val = v
                    entry: dict = {"categoria": code_norm}
                    if acc_desc:
                        entry["descrizione"] = _clean(acc_desc)[:200]
                    if imp_val:
                        entry["importo_minimo"] = imp_val
                    if grado_val:
                        entry["grado_complessita"] = grado_val
                    if not any(c.get("categoria") == code_norm for c in _bold_cats):
                        _bold_cats.append(entry)
                    acc_desc = ""
                else:
                    cleaned = re.sub(r'\*+', '', stripped).strip()
                    if cleaned and len(cleaned) > 5 and not re.match(r'^[\d.,€Ç\s]+$', cleaned):
                        if "TOTALE" in cleaned.upper() or "CATEGORIA" in cleaned.upper():
                            acc_desc = ""
                        elif re.match(
                            r"(?:EDILIZIA|IMPIANTI|STRUTTUR|IDRAULIC|VIABILIT|INFRASTRUTTUR)",
                            cleaned, re.IGNORECASE,
                        ):
                            acc_desc = cleaned
                        elif acc_desc:
                            acc_desc = (acc_desc + " " + cleaned).strip()

        if _bold_cats:
            if "servizi_analoghi" not in srv:
                srv["servizi_analoghi"] = {"richiesti": True}
            srv["servizi_analoghi"]["categorie_richieste"] = _bold_cats

    # ── Arricchisci servizi_analoghi ─────────────────────────────────────
    sa = srv.get("servizi_analoghi", {})
    if sa.get("richiesti") and sa.get("categorie_richieste"):
        if not sa.get("numero_servizi"):
            m_num = re.search(
                r"n\.\s*(\d+)\s*(?:\([^)]*\)\s*)?servizi", text, re.IGNORECASE,
            )
            if m_num:
                sa["numero_servizi"] = int(m_num.group(1))
            else:
                m_num2 = re.search(r"max\s+di\s+n\.\s*(\d+)\s+servizi", text, re.IGNORECASE)
                if m_num2:
                    sa["numero_servizi"] = int(m_num2.group(1))
        if not sa.get("periodo_riferimento"):
            m_per = re.search(
                r"(?:nei|negli)\s+((?:ultimi\s+)?\w+\s+anni\s+antecedent[^.,;]{0,100})",
                text, re.IGNORECASE,
            )
            if m_per:
                sa["periodo_riferimento"] = _clean(m_per.group(1))[:150]
        if not sa.get("tipologia"):
            m_tip = re.search(
                r"n\.\s*\d+\s*(?:\([^)]*\)\s*)?servizi\s+di\s+([^,]{5,80}?)(?:\s*[*]*\s*,\s*relativ|\s*[*]*\s*\.)",
                text, re.IGNORECASE,
            )
            if m_tip:
                sa["tipologia"] = _clean(re.sub(r'\*+', '', m_tip.group(1)))[:100]
        if not sa.get("note"):
            m_unico = re.search(
                r"(?:possibil\w+\s+)?(?:dimostrare\s+il\s+possesso\s+(?:del\s+)?presente\s+requisit\w+\s+)?(?:anche\s+)?mediante\s+"
                r"un\s+unico\s+servizio\s*[,\s]*(?:purché|a\s+condizione\s+che|purchÚ)[^.]{0,200}\.",
                text, re.IGNORECASE,
            )
            if m_unico:
                sa["note"] = _clean(m_unico.group(0))[:300]
            else:
                m_unico2 = re.search(
                    r"in\s+luogo\s+dei\s+due\s+servizi[^.]{0,300}\.",
                    text, re.IGNORECASE,
                )
                if m_unico2:
                    sa["note"] = _clean(m_unico2.group(0))[:300]

    # ── Personale tecnico medio ──────────────────────────────────────────
    m_pers = re.search(
        r"(?:organico|personale)\s+(?:tecnico\s+)?medio[^.]{0,100}?(\d+)\s*(?:unit[àa]|dipendent)",
        text, re.IGNORECASE,
    )
    if m_pers:
        srv["personale_tecnico_medio"] = {
            "richiesto": True,
            "numero_minimo": int(m_pers.group(1)),
        }

    # ── Gruppo di lavoro – figure professionali ──────────────────────────
    gdl = rp["gruppo_di_lavoro"]
    figure: list[dict] = []

    # Strategia 1: tabella pipe-delimited (GRUPPO DI LAVORO)
    gdl_table_section = _section_text(
        text,
        ["GRUPPO DI LAVORO", "STRUTTURA OPERATIVA MINIMA", "COMPOSIZIONE DEL GRUPPO",
         "FIGURE PROFESSIONALI RICHIESTE", "Prestazione/Figura professionale"],
        ["AVVALIMENTO", "SUBAPPALTO", "GARANZI", "CRITERI", "CAPACIT", "FATTURATO",
         "Il Gruppo di lavoro dovrà", "Il gruppo di lavoro dovrà",
         "9.", "10.", "11.", "SERVIZI DI PUNTA", "SERVIZI ANALOGHI"],
        max_len=10000,
    )
    if gdl_table_section:
        table_rows: list[list[str]] = []
        current_row_cells: list[str] | None = None
        for line in gdl_table_section.split("\n"):
            stripped = line.strip()
            if not stripped or "|" not in stripped:
                continue
            raw_cells = stripped.split("|")
            cells = [c.strip() for c in raw_cells]
            while cells and cells[0] == "":
                cells.pop(0)
            while cells and cells[-1] == "":
                cells.pop()
            if not cells:
                continue
            first = cells[0]
            if re.match(r"^\d+$", first):
                if current_row_cells is not None:
                    table_rows.append(current_row_cells)
                current_row_cells = cells
            elif current_row_cells is not None:
                for ci in range(len(cells)):
                    target = ci + 1
                    if target < len(current_row_cells) and cells[ci]:
                        if current_row_cells[target]:
                            current_row_cells[target] += " " + cells[ci]
                        else:
                            current_row_cells[target] = cells[ci]
        if current_row_cells is not None:
            table_rows.append(current_row_cells)

        for row in table_rows:
            ruolo_raw = row[1] if len(row) > 1 else ""
            requisiti_raw = row[2] if len(row) > 2 else ""
            if len(row) > 3:
                requisiti_raw += " " + " ".join(row[3:])
            ruolo_clean = _clean(ruolo_raw)
            req_clean = _clean(requisiti_raw)
            if ruolo_clean and len(ruolo_clean) > 5:
                entry: dict = {"ruolo": ruolo_clean}
                if req_clean and len(req_clean) > 3:
                    entry["requisiti"] = req_clean
                figure.append(entry)

    # ── Strategia 1b: tabella plain-text interleaved ─────────────────────
    if not figure and gdl_table_section:
        has_table_header = bool(re.search(
            r"(?:Prestazione|Figura\s+professionale|Ruolo).*?(?:Requisit|Titol)",
            gdl_table_section, re.IGNORECASE | re.DOTALL,
        ))
        has_roles = bool(re.search(
            r"Tecnico\s+esperto|Progettist\w+\s+(?:architetton|struttur|impiantist)",
            gdl_table_section, re.IGNORECASE,
        ))
        if has_table_header or has_roles:
            hdr_m = re.search(
                r"(?:Prestazione.*?Requisit\w*|Figura\s+professionale.*?Requisit\w*)",
                gdl_table_section, re.IGNORECASE | re.DOTALL,
            )
            body = gdl_table_section[hdr_m.end():] if hdr_m else gdl_table_section

            _req_split = re.compile(
                r"((?:Laurea\s+(?:magistrale|triennale)|Diploma\s|Abilitazione\s|"
                r"iscrizione\s+al\s+rispettivo)\b)",
                re.IGNORECASE,
            )
            _role_start = re.compile(
                r"^(?:Tecnico\s+esperto\s+(?:in|di)\b|"
                r"Progettist\w+\s+(?:architetton|struttur|impiantist|coordinat)|"
                r"Coordinatore\s+(?:per\s+la|della\s+)sicurezza|"
                r"Direttore\s+(?:dei\s+lavori|operativo|tecnico)|"
                r"Geologo\b|Collaudator\w+\s|"
                r"Ispettor\w+\s+di\s+cantiere|"
                r"Professionista\s+antincendio|"
                r"Esperto\s+(?:ambientale|CAM)|"
                r"BIM\s+(?:manager|coordinator|specialist)|Topografo\b)",
                re.IGNORECASE,
            )

            role_frags: list[str] = []
            req_frags: list[str] = []
            entries: list[dict] = []

            for line in body.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                stripped = re.sub(r"---\s*Pagina\s+\d+\s*---", "", stripped).strip()
                stripped = re.sub(r"\[Pagina\s+\d+\]", "", stripped).strip()
                if not stripped:
                    continue
                if re.match(r"^\d{1,2}$", stripped):
                    continue
                stripped = re.sub(r"^(\d{1,2})\s+", "", stripped)
                if _role_start.match(stripped) and (role_frags or req_frags):
                    entries.append({"rf": list(role_frags), "qf": list(req_frags)})
                    role_frags.clear()
                    req_frags.clear()
                parts = _req_split.split(stripped)
                if len(parts) > 1:
                    left = parts[0].strip()
                    right = "".join(parts[1:]).strip()
                    if left:
                        role_frags.append(left)
                    if right:
                        req_frags.append(right)
                else:
                    if stripped.lower().startswith(("iscrizione", "laurea", "diploma", "abilitazione")):
                        req_frags.append(stripped)
                    else:
                        role_frags.append(stripped)

            if role_frags or req_frags:
                entries.append({"rf": list(role_frags), "qf": list(req_frags)})

            for idx_e, e in enumerate(entries, 1):
                role = _clean(" ".join(e["rf"]))
                req = _clean(" ".join(e["qf"]))
                if role and len(role) > 5:
                    entry = {"ruolo": role, "numero": idx_e}
                    if req and len(req) > 3:
                        entry["requisiti"] = req
                    figure.append(entry)

    # ── Strategia 2 (fallback): regex su testo libero ────────────────────
    if not figure:
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
            (r"(?:tecnico\s+esperto\s+(?:in|di)\s+[^\n]{5,200}?)(?=\n|\s{2,}|Laurea)", None),
        ]
        for pat, _ in ruoli_patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            for m in matches:
                role = _clean(m)
                if not role:
                    continue
                rl = role.lower()
                is_dup = any(
                    rl == ex or rl[:40] == ex[:40] or rl.startswith(ex[:40]) or ex.startswith(rl[:40])
                    or rl in ex or ex in rl
                    for ex in (f.get("ruolo", "").lower() for f in figure)
                )
                if is_dup:
                    continue
                entry = {"ruolo": role}
                idx_r = text_lower.find(role.lower())
                if idx_r >= 0:
                    ctx = text[idx_r:idx_r + 500]
                    m_req_r = re.search(r"(?:abilitazione|iscrizione|iscritt\w+)\s+([^.]{10,300})", ctx, re.IGNORECASE)
                    if m_req_r:
                        entry["requisiti"] = _clean(m_req_r.group(0))
                entry["_pos"] = idx_r
                figure.append(entry)

    # Post-dedup
    if len(figure) > 1:
        to_remove: set[int] = set()
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
        fig_section = req_section or _section_text(
            text,
            ["GRUPPO DI LAVORO", "FIGURE PROFESSIONALI", "COMPOSIZIONE DEL GRUPPO", "STRUTTURA OPERATIVA",
             "TEAM DI PROGETTAZIONE", "SOGGETTI RICHIESTI"],
            ["AVVALIMENTO", "SUBAPPALTO", "GARANZI", "CRITERI", "9.", "10."],
            max_len=10000,
        )
        if fig_section:
            fig_list_patterns = [
                r"n\.?\s*°?\s*(\d+)\s+((?:[A-Z][a-z]+\s+){1,5}[a-z]+(?:\s+[a-z]+){0,3})",
                r"(?:^|\n)\s*(?:\d+|[a-h])\s*[.)]\s*([A-Z][^\n]{10,120}?)(?:\n|;)",
                r"(?:^|\n)\s*[-•–]\s*([A-Z][^\n]{10,120}?)(?:\n|;)",
                r"(?:^|\n)\s*(\d)\s+(Tecnico\s+[^\n]{10,120}?)(?=\s+Laurea|\s+Diploma|\s+Abilit|\n\n)",
            ]
            for fig_pat in fig_list_patterns:
                fig_matches = re.findall(fig_pat, fig_section, re.MULTILINE)
                if fig_matches and len(fig_matches) >= 2:
                    for fm in fig_matches:
                        role_text = fm[-1] if isinstance(fm, tuple) else fm
                        role_text = _clean(role_text)
                        if role_text and len(role_text) > 5:
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

    # ── Cumulabilità ruoli ───────────────────────────────────────────────
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
    non_cumul = re.search(
        r"(?:non\s+(?:sono\s+)?cumulabil\w+|non\s+(?:è\s+)?(?:consentit\w+|ammess\w+)\s+il\s+cumulo|"
        r"incompatibil\w+[^.]{0,60}?ruol\w+)",
        text, re.IGNORECASE,
    )
    if cumul_found or non_cumul:
        gdl["ruoli_cumulabili"] = cumul_found and not non_cumul

    # ── Numero minimo professionisti ─────────────────────────────────────
    m_num_min = re.search(
        r"(?:gruppo\s+di\s+lavoro|struttura\s+operativa)[^.]{0,200}?"
        r"(?:composto\s+(?:da\s+)?(?:almeno\s+|minimo\s+)?|"
        r"almeno\s+|minimo\s+|non\s+inferiore\s+a\s+)"
        r"(\d+)\s*(?:professionisti|componenti|soggetti|unit[àa])",
        text, re.IGNORECASE | re.DOTALL,
    )
    if not m_num_min:
        if len(gdl.get("figure_professionali", [])) >= 2:
            gdl["numero_minimo_professionisti"] = len(gdl["figure_professionali"])
    else:
        gdl["numero_minimo_professionisti"] = int(m_num_min.group(1))

    return rp
