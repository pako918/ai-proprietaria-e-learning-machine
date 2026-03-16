"""Sezione D: Lotti e importi."""

import re
from .utils import _parse_euro, _clean


def extract_lotti(text: str, text_lower: str, ig: dict) -> tuple[dict, dict]:
    """Estrae lotti, importi, categorie.

    Returns:
        (sl, ic) — suddivisione_lotti e importi_complessivi.
        Nota: modifica ig in-place per aggiungere categorie_trovate.
    """
    sl = {"lotti": []}
    ic = {}

    # Numero lotti
    lotti_nums = re.findall(r"lotto\s+(?:n\.?\s*)?(\d+)", text, re.IGNORECASE)
    if lotti_nums:
        n_lotti = max(int(n) for n in lotti_nums)
        if n_lotti > 50:
            n_lotti = 1
        sl["numero_lotti"] = n_lotti
    else:
        sl["numero_lotti"] = 1
        if "lotto unico" in text_lower or "unico lotto" in text_lower or "non suddivisa in lotti" in text_lower:
            sl["lotto_unico_motivazione"] = "Lotto unico come da disciplinare"

    # Importo totale complessivo
    imp_patterns = [
        r"importo\s+globale[^€Ç\d]{0,100}?(?:pari\s+ad?\s+)?(?:€|Ç|Euro|euro|EUR)\s*\.?\s*([\d.,]+)",
        r"importo\s+complessivo\s+dell['\u2019\s]\s*appalto[^€Ç\d]{0,80}?(?:€|Ç|Euro|euro|EUR)\s*\.?\s*([\d.,]+)",
        r"importo\s+complessivo\s+(?:Euro|€|di\s+€)\s*([\d.,]+)",
        r"importo\s+complessivo[^€Ç\d]{0,40}?(?:Euro|euro|EUR)\s*\.?\s*([\d.,]+)",
        r"importo\s+complessivo[^€Ç\d]{0,80}?(?:pari\s+ad?|ammonta\s+a)\s*(?:€|Ç|Euro|euro|EUR)\s*\.?\s*([\d.,]+)",
        r"importo\s+complessivo[^€Ç\d]{0,120}?(?:di|pari\s+a|per)\s+(?:€|Ç|Euro|euro|EUR)\s*\.?\s*([\d.,]+)",
        r"(?:base\s+di\s+gara|base\s+d['\u2019]asta)[^€Ç\d]{0,60}?(?:pari\s+ad?|ammonta\s+a)\s*(?:€|Ç|Euro|euro|EUR)\s*\.?\s*([\d.,]+)",
        r"importo\s+(?:a\s+)?base\s+(?:di\s+|d['\u2019])?(?:gara|asta)\s*[€Ç:]\s*\.?\s*([\d.,]+)",
        r"(?:^|[\n.])(?:[^%\n]{0,20})importo\s+(?:a\s+)?base\s+(?:di\s+|d['\u2019])?(?:gara|asta)[^€Ç\d]{0,40}?(?:Euro|euro|EUR)\s*\.?\s*([\d.,]+)",
        r"STIMA\s+TOTALE\s+DEL\s+VALORE[^€Ç\d]{0,80}?(?:€|Ç|Euro|euro|EUR)\s*\.?\s*([\d.,]+)",
        r"valore\s+(?:del(?:la)?\s+)?(?:contratto\s+di\s+)?concessione[^€Ç\d]{0,80}?(?:€|Ç|euro|Euro)\s*\.?\s*([\d.,]+)",
        r"valore\s+(?:del(?:la)?\s+)?concessione[^€Ç\d]{0,80}?(?:pari\s+a|determinat\w+\s+in)\s*[:\s]*(?:€|Ç|euro|Euro)\s*\.?\s*([\d.,]+)",
        r"(?:determinat\w+\s+in|pari\s+a)\s*[:\s]*euro\s+([\d.,]+(?://\d{2})?)",
        r"importo\s+totale\s+(?:stimat\w+\s+)?(?:dell['\u2019]appalto\s+)?(?:€|Euro|:)?\s*(?:pari\s+a\s+)?(?:€\s*)?([\d.,]+)",
        r"valore\s+globale\s+dell['\u2019]?\s*appalto[^\d]{0,60}?([\d.,]+)",
        r"valore\s+massimo\s+stimato[^\d]{0,80}?(?:Euro|€)\s*([\d.,]+)",
        r"importo\s+complessivo[^€Ç\d]{0,80}?[€Ç]\s*\.?\s*([\d.,]+)",
        r"importo\s+dell['\u2019]?\s*appalto\s*\n?\s*(?:€|Euro)\s*([\d.,]+)",
        r"importo\s+dell['\u2019]?\s*appalto\s*[:\s]*(?:€|Euro)\s*([\d.,]+)",
        r"(?:fissato|stabilito)\s+in\s*(?:€|Ç|Euro|euro)\s*\.?\s*([\d.,]+)",
        r"TOTALE\s*[€Ç]\s*\.?\s*([\d.,]+)",
        r"(?:importo\s+complessivo|importo\s+totale|base\s+di\s+gara|base\s+d['\u2019]asta)[^.]{0,100}?([\d.,]+)\s*(?:€|Euro|euro|EUR)",
    ]
    for pat in imp_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            raw_val = m.group(1)
            raw_val = re.sub(r'//\d{2}$', '', raw_val)
            val = _parse_euro(raw_val)
            if val and val > 100:
                ic["importo_totale_gara"] = val
                break

    # Importo soggetto a ribasso
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

    # Costo manodopera (quota fissa non soggetta a ribasso)
    mano_patterns = [
        r"(?:costo|costi)\s+della\s+manodopera[^€Ç\d]{0,80}?(?:€|Ç|euro)?\s*\.?\s*([\d.,]+)",
        r"quota\s+(?:non\s+soggett\w+\s+a\s+ribasso|fissa)[^€Ç\d]{0,80}?(?:€|Ç|euro)?\s*\.?\s*([\d.,]+)",
        r"manodopera[^€Ç\d]{0,60}?(?:€|Ç|euro)\s*\.?\s*([\d.,]+)",
    ]
    for _mp in mano_patterns:
        m_mano = re.search(_mp, text, re.IGNORECASE)
        if m_mano:
            v = _parse_euro(m_mano.group(1))
            if v and v > 100:
                ic["costo_manodopera"] = v
                break

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
        _fisso = bool(re.search(
            r"fisso\s+e\s+invariabile|non\s+soggetto\s+a\s+(?:variazioni?|revisione)",
            text, re.IGNORECASE,
        ))
        if _fisso:
            rev = {"ammessa": False, "note": "Corrispettivo fisso e invariabile"}
            m_art60 = re.search(r"art\.?\s*60[^.]{0,100}", text, re.IGNORECASE)
            if m_art60:
                rev["riferimento_normativo"] = _clean(m_art60.group(0))[:120]
            ic["revisione_prezzi"] = rev
        else:
            rev = {"ammessa": True}
            m_rev_perc = re.search(r"(?:revisione[^.]{0,200}?)(\d{1,3})\s*%\s*(?:dell['\u2019])?(?:variazione|incremento)", text, re.IGNORECASE)
            if not m_rev_perc:
                m_rev_perc = re.search(r"(?:soglia|superiore)\s+(?:al\s+)?(\d{1,3})\s*%", text[text_lower.find("revisione"):text_lower.find("revisione")+500], re.IGNORECASE)
            if m_rev_perc:
                rev["soglia_percentuale"] = int(m_rev_perc.group(1))
            ic["revisione_prezzi"] = rev

    # --- Luogo di esecuzione (a livello di documento) ---
    luogo_esecuzione = None
    _luogo_patterns = [
        r"(?:luogo\s+(?:di\s+)?(?:esecuzione|svolgimento|consegna|prestazione|fornitura))[\s:–\-]*\n?\s*([^\n]{5,120})",
        r"(?:II\.1\.4|I\.4|I\.4\))\s*[:\-]?\s*(?:luogo\s+(?:di\s+)?(?:esecuzione|svolgimento|consegna|prestazione))?[\s:]*\n?\s*([^\n]{5,120})",
        r"(?:NUTS\s+code|codice\s+NUTS)[^\n]{0,60}\n\s*([A-Za-zÀ-ùÀ-ú][^\n]{5,100})",
        r"(?:Comune|Città|Municipio|Provincia)\s+di\s+([A-Za-zÀ-ùÀ-ú][^\n,;.]{2,60})\s*(?:\([A-Z]{2}\))",
    ]
    _bad_luogo_markers = ("http", "decreto", "prot.", "pec", "@", "piattaforma", "telematic", "traspare", "acquistinrete")
    for _lp in _luogo_patterns:
        _m = re.search(_lp, text, re.IGNORECASE)
        if _m:
            _v = _clean(_m.group(1))
            if _v and len(_v) <= 150 and not any(b in _v.lower() for b in _bad_luogo_markers):
                luogo_esecuzione = _v
                break

    # --- Dettaglio per singoli lotti ---
    for lotto_n in range(1, sl["numero_lotti"] + 1):
        lotto_data = {"numero": lotto_n}

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
                next_lotto = re.search(
                    f"(?:lotto\\s*(?:n\\.?\\s*)?{lotto_n + 1}\\b|^\\d+\\.\\s+[A-Z])",
                    text[candidate + 20:],
                    re.IGNORECASE | re.MULTILINE,
                )
                segment_len = next_lotto.start() + 20 if next_lotto else 8000
                if segment_len >= 200:
                    lotto_start = candidate
                    break
                search_from = candidate + len(m.group(0))
            if lotto_start >= 0:
                break

        if lotto_start >= 0:
            next_lotto = re.search(
                f"(?:lotto\\s*(?:n\\.?\\s*)?{lotto_n + 1}\\b|^\\d+\\.\\s+[A-Z])",
                text[lotto_start + 20:],
                re.IGNORECASE | re.MULTILINE,
            )
            lotto_end = lotto_start + (next_lotto.start() + 20 if next_lotto else 8000)
            lotto_text = text[lotto_start:lotto_end]

            first_lines = lotto_text[:500]
            m_desc = re.search(r"(?:lotto\s*(?:n\.?\s*)?\d+)\s*[:\-\n]\s*(.+?)(?:\n\n|\n(?:Tabella|IMPORTO|Prestazione))", first_lines, re.IGNORECASE | re.DOTALL)
            if m_desc:
                lotto_data["denominazione"] = _clean(m_desc.group(1))

            imp_lotto = re.search(r"(?:IMPORTO\s+LOTTO|importo\s+(?:del\s+)?lotto)[^€Ç\d]{0,30}[€Ç\s]*([\d.,]+)", lotto_text, re.IGNORECASE)
            if imp_lotto:
                v = _parse_euro(imp_lotto.group(1))
                if v:
                    lotto_data["importo_base_asta"] = v

            imp_base = re.search(r"[Ii]mporto\s+totale\s+a\s+base\s+di\s+gara\s*[:\n]\s*[€Ç]?\s*([\d.,]+)", lotto_text)
            if imp_base:
                v = _parse_euro(imp_base.group(1))
                if v:
                    lotto_data["importo_base_asta"] = v

            if "importo_base_asta" not in lotto_data:
                imp_valore = re.search(
                    r"[Vv]alore\s+appalto\s*(?:complessivo)?\s*€\s*([\d.,]+)",
                    lotto_text, re.IGNORECASE,
                )
                if imp_valore:
                    v = _parse_euro(imp_valore.group(1))
                    if v and v > 100:
                        lotto_data["importo_base_asta"] = v

            if "importo_base_asta" not in lotto_data:
                imp_contr = re.search(
                    r"importo\s+contrattuale\s+(?:complessivo\s+)?pari\s+a\s*€?\s*([\d.,]+)",
                    lotto_text, re.IGNORECASE,
                )
                if imp_contr:
                    v = _parse_euro(imp_contr.group(1))
                    if v and v > 100:
                        lotto_data["importo_base_asta"] = v

            m_rib_l = re.search(r"soggett\w+\s+a\s+ribasso[^€Ç\d]{0,50}[€Ç\s]*([\d.,]+)", lotto_text, re.IGNORECASE)
            if m_rib_l:
                v = _parse_euro(m_rib_l.group(1))
                if v:
                    lotto_data["importo_soggetto_ribasso"] = v

            m_perc = re.search(r"(\d{1,3})\s*%\s*(?:di\s+[A-Z]|del\s+corrispettivo)", lotto_text, re.IGNORECASE)
            if m_perc:
                p = int(m_perc.group(1))
                if 1 <= p <= 100:
                    lotto_data["quota_ribassabile_percentuale"] = p

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

            m_dur = re.search(r"durat\w+[^.]{0,80}?(\d+)\s*(?:giorni|gg)", lotto_text, re.IGNORECASE)
            if m_dur:
                lotto_data["durata_esecuzione"] = {"giorni": int(m_dur.group(1))}
            else:
                m_dur = re.search(r"durat\w+[^.]{0,80}?(\d+)\s*mesi", lotto_text, re.IGNORECASE)
                if m_dur:
                    lotto_data["durata_esecuzione"] = {"mesi": int(m_dur.group(1))}

        if luogo_esecuzione and "luogo_esecuzione" not in lotto_data:
            lotto_data["luogo_esecuzione"] = luogo_esecuzione
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
        if "denominazione" not in lotto:
            n = lotto["numero"]
            m_den = re.search(
                rf"Lotto\s*(?:n\.?\s*)?{n}\s*[–\-\u2013]\s*(.{{10,200}}?)(?:\n|ID\s|CATEGORIA|\d{{1,2}}\.\d)",
                text, re.IGNORECASE,
            )
            if m_den:
                lotto["denominazione"] = _clean(m_den.group(1))

    if sl["numero_lotti"] == 1 and sl["lotti"]:
        lotto = sl["lotti"][0]
        if "importo_base_asta" not in lotto and ic.get("importo_totale_gara"):
            lotto["importo_base_asta"] = ic["importo_totale_gara"]
        if "importo_soggetto_ribasso" not in lotto and ic.get("importo_totale_soggetto_ribasso"):
            lotto["importo_soggetto_ribasso"] = ic["importo_totale_soggetto_ribasso"]
        if "denominazione" not in lotto and ig.get("titolo"):
            lotto["denominazione"] = ig["titolo"]
        if luogo_esecuzione:
            lotto["luogo_esecuzione"] = luogo_esecuzione
    elif sl["numero_lotti"] == 1 and not sl["lotti"]:
        lotto = {"numero": 1}
        if ic.get("importo_totale_gara"):
            lotto["importo_base_asta"] = ic["importo_totale_gara"]
        if ig.get("titolo"):
            lotto["denominazione"] = ig["titolo"]
        if luogo_esecuzione:
            lotto["luogo_esecuzione"] = luogo_esecuzione
        sl["lotti"].append(lotto)

    # --- Estrazione lotti da tabella multi-colonna (formato QTE) ---
    _euro_sym_pat = r"[€Ç]"
    _amt_line_pat = (
        r"((?:\d{1,3}[\s.])*\d{1,3}(?:[.,]\d{1,2})?|\d{4,}(?:[.,]\d{1,2})?)\s*"
        + _euro_sym_pat
    )
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
            if _stripped.upper().startswith("TOTALE"):
                continue
            if _stripped.startswith("[TABELLA"):
                continue
            _amounts_raw = re.findall(_amt_line_pat, _qline)
            if len(_amounts_raw) >= 2:
                _netto_raw = _amounts_raw[-2].replace(" ", "")
                _v_netto = _parse_euro(_netto_raw)
                if _v_netto and _v_netto > 50000:
                    lotti_from_qte.append(_v_netto)

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

    # Categorie globali
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

    cats_simple = re.findall(r"(?:categori\w+|class\w+)\s+((?:[ESDIV]\.?\d{2}(?:\s*[-,e]\s*)?)+)", text, re.IGNORECASE)
    if cats_simple:
        found_cats = set()
        for match in cats_simple:
            for c in re.findall(r"[ESDIV]\.?\d{2}", match):
                found_cats.add(c[:1] + "." + c[-2:] if "." not in c else c)
        ig["categorie_trovate"] = sorted(found_cats)

    og_os_matches = re.findall(r"\b(O[GS]\d{1,2})\b", text)
    if og_os_matches:
        unique_og_os = list(dict.fromkeys(og_os_matches))
        if "categorie_trovate" not in ig:
            ig["categorie_trovate"] = []
        for cat in unique_og_os:
            if cat not in ig["categorie_trovate"]:
                ig["categorie_trovate"].append(cat)

    return sl, ic
