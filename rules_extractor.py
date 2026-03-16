"""
AppaltoAI — Estrattore Deterministico (Regole + Regex)
Nessun modello ML coinvolto in questa fase.
Responsabilità unica: estrarre dati strutturati da testo con regex.
"""

import re
from typing import Optional, Tuple

from utils import (
    clean_string, normalize_amount, parse_number_word,
    find_value_context, first_match, all_matches, extract_int,
)
from field_registry import registry


class RulesExtractor:
    """Estrazione puramente deterministica basata su regex.
    Nessun modello ML coinvolto in questa fase."""

    def __init__(self):
        self._patterns = registry.get_patterns()

    def reload_patterns(self):
        self._patterns = registry.get_patterns()

    # ── Utility (delegano a utils.py per evitare duplicazione) ────

    @staticmethod
    def clean(s: str) -> str:
        return clean_string(s)

    @staticmethod
    def normalize_amount(raw) -> Optional[str]:
        return normalize_amount(raw)

    def first_match(self, text: str, patterns: list) -> Optional[str]:
        return first_match(text, patterns)

    def all_matches(self, text: str, patterns: list) -> list:
        return all_matches(text, patterns)

    def extract_int(self, text: str, patterns: list) -> Optional[int]:
        return extract_int(text, patterns)

    def parse_number_word(self, s) -> Optional[int]:
        return parse_number_word(s)

    # ── Estrazioni complesse ───────────────────────────────────────

    def extract_lotti_detail(self, text: str) -> list:
        lotti = []
        parts = re.split(r'(?i)\bLotto\s+(\d+)\b', text)
        if len(parts) > 2:
            for i in range(1, len(parts) - 1, 2):
                n = parts[i]; body = parts[i + 1][:2000]
                cig_m = re.search(r'\bCIG[:\s]*([A-Z0-9]{10})\b', body, re.I)
                cup_m = re.search(r'\bCUP[:\s]*([A-Z][0-9]{2}[A-Z][0-9]{8})\b', body, re.I)
                imp_m = re.search(r'(?:importo|base\s+gara)[:\s€]*([0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})?)', body, re.I)
                desc_m = re.search(r'^(.{10,200})', body.strip())
                lotti.append({
                    "numero": int(n),
                    "cig": cig_m.group(1) if cig_m else None,
                    "cup": cup_m.group(1) if cup_m else None,
                    "importo": normalize_amount(imp_m.group(1)) if imp_m else None,
                    "descrizione": clean_string(desc_m.group(1))[:200] if desc_m else None,
                })
        return lotti

    def extract_categorie_gara(self, text: str) -> list:
        """Estrae categorie strutturate con codice, descrizione, importo e tipo."""
        categorie = []
        seen = set()
        for m in re.finditer(
            r'\b(E\.?\d{2}|S\.?\d{2}|IA\.?\d{2}|IB\.?\d{2}|D\.?\d{2})\b'
            r'[^\n]{0,100}?'
            r'(?:€?\s*([0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})?))?',
            text, re.I
        ):
            codice = m.group(1).upper()
            codice = re.sub(r'^(IA|IB|[ESDG])(\d+)$', lambda x: x.group(1) + '.' + x.group(2), codice)
            if codice in seen:
                continue
            seen.add(codice)
            importo = normalize_amount(m.group(2)) if m.group(2) else None
            ctx = text[max(0, m.start()-10):m.end()+200]
            desc_m = re.search(re.escape(m.group(0)) + r'\s*[-–:]\s*([^\n€]{5,120})', ctx, re.I)
            descrizione = clean_string(desc_m.group(1))[:120] if desc_m else None
            categorie.append({
                "codice": codice,
                "descrizione": descrizione,
                "importo": importo,
                "tipo": "SOA" if codice.startswith(("OG", "OS")) else "Ingegneria",
            })
        for m in re.finditer(
            r'\b(OG\.?\d{1,2}|OS\.?\d{1,2})\b'
            r'[^\n]{0,100}?'
            r'(?:€?\s*([0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})?))?',
            text, re.I
        ):
            codice = m.group(1).upper()
            codice = re.sub(r'^(OG|OS)(\d+)$', lambda x: x.group(1) + '.' + x.group(2), codice)
            if codice in seen:
                continue
            seen.add(codice)
            importo = normalize_amount(m.group(2)) if m.group(2) else None
            ctx = text[max(0, m.start()-10):m.end()+200]
            desc_m = re.search(re.escape(m.group(0)) + r'\s*[-–:]\s*([^\n€]{5,120})', ctx, re.I)
            descrizione = clean_string(desc_m.group(1))[:120] if desc_m else None
            categorie.append({
                "codice": codice,
                "descrizione": descrizione,
                "importo": importo,
                "tipo": "SOA",
            })
        return categorie

    def extract_figure_professionali(self, text: str) -> list:
        """Estrae figure professionali richieste con dettagli."""
        figure = []
        for m in re.finditer(
            r'(?:figura|professionista|ruolo)\s*(?:\d+)?[:\s-]*\s*'
            r'([^\n]{5,150})',
            text, re.I
        ):
            block = m.group(1).strip()
            nome = clean_string(block.split(',')[0].split('–')[0].split('-')[0])[:80]
            if len(nome) < 3:
                continue
            exp_m = re.search(r'(\d+)\s*ann[io]\s*(?:di\s+)?esperienza', block, re.I)
            esperienza = int(exp_m.group(1)) if exp_m else None
            laurea_m = re.search(r'(?:laurea|diploma)\s+(?:in\s+)?([^\n,;]{5,60})', block, re.I)
            laurea = clean_string(laurea_m.group(1))[:60] if laurea_m else None
            albo_m = re.search(r'(?:iscri(?:tto|zione)\s+(?:all\'?\s*)?albo|abilitazione)', block, re.I)
            figure.append({
                "nome": nome,
                "laurea": laurea,
                "iscrizione_albo": bool(albo_m),
                "esperienza_anni": esperienza,
            })
        return figure[:20]

    def extract_categorie_ingegneria(self, text: str) -> list:
        cats = set()
        for m in re.finditer(r'\b(E\.?\d{2}|S\.?\d{2}|IA\.?\d{2}|IB\.?\d{2}|D\.?\d{2})\b', text, re.I):
            cat = m.group(1).upper()
            cat = re.sub(r'^(IA|IB|[ESDG])(\d+)$', lambda x: x.group(1) + '.' + x.group(2), cat)
            cats.add(cat)
        return sorted(cats)

    def extract_criteri_tecnici(self, text: str) -> list:
        criteri = []
        for m in re.finditer(
            r'(?:criterio\s+)?([A-Z](?:\.\d+)?)\s*[-.:]\s*([^\n]{5,120}?)\s+(\d+)\s*(?:punti?|pt|p\.ti)',
            text, re.I | re.M
        ):
            criteri.append({"codice": m.group(1), "nome": self.clean(m.group(2))[:100], "punteggio": int(m.group(3))})
        return criteri[:15]

    def extract_struttura_compenso(self, text: str) -> dict:
        result = {
            "quota_fissa_65_perc": bool(re.search(r'65\s*%[^\n]{0,60}(?:fisso|non\s+soggett[oa]|prezzo\s+fisso)', text, re.I)),
            "quota_ribassabile_35_perc": bool(re.search(r'35\s*%[^\n]{0,60}(?:ribassabile|soggett[oa]|disponibile)', text, re.I)),
        }
        m35 = re.search(r'35\s*%[^\n]{0,60}?€?\s*([0-9]{1,3}(?:[.,][0-9]{3})*[.,]\d{2})', text, re.I)
        if m35:
            result["importo_ribassabile"] = self.normalize_amount(m35.group(1))
        return result

    def extract_scadenze(self, text: str) -> dict:
        scadenze = {}
        main = self.first_match(text, self._patterns.get("scadenza_offerte", []))
        if main:
            scadenze["principale"] = main
        m_fase = re.search(r'FASE\s+1[^\n]{0,40}?(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}[^\n]{0,30})', text, re.I)
        if m_fase:
            scadenze["fase_1_archivi"] = self.clean(m_fase.group(1))
        m_pub = re.search(r'(?:prima\s+)?sessione\s+pubblica[^\n]{0,30}?(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}[^\n]{0,20})', text, re.I)
        if m_pub:
            scadenze["sessione_pubblica"] = self.clean(m_pub.group(1))
        return scadenze

    def extract_vincoli_lotti(self, text: str) -> dict:
        vincoli = {
            "partecipazione_tutti_lotti": bool(re.search(r'(?:obbligo|vincolo)\s+di\s+partecipazione.{0,50}(?:entrambi|tutti)', text, re.I)),
            "offerta_identica": bool(re.search(r'offerta\s+identica.{0,30}(?:entrambi|tutti)', text, re.I)),
            "max_lotti_aggiudicazione": None,
            "medesima_forma_giuridica": bool(re.search(r'medesima\s+forma\s+giuridica', text, re.I)),
        }
        m = re.search(r'(?:massimo|max)\s+(\d+)\s+lott[io]\s+(?:per\s+)?(?:concorrente|aggiudicatario)', text, re.I)
        if m:
            vincoli["max_lotti_aggiudicazione"] = int(m.group(1))
        return vincoli

    def extract_sopralluogo(self, text: str) -> dict:
        obbligatorio = bool(re.search(r'sopralluogo\s+(?:e\s+)?obbligatorio|sopralluogo\s+delle?\s+aree', text, re.I))
        non_previsto = bool(re.search(r'sopralluogo[^\n]{0,30}non\s+(?:prev|richiesto)', text, re.I))
        m = re.search(r'sopralluogo[^\n]{0,300}', text, re.I)
        return {"obbligatorio": obbligatorio and not non_previsto, "note": self.clean(m.group(0))[:250] if m else None}

    def extract_note_operative(self, text: str) -> list:
        note = []
        checks = [
            (r'sopralluogo\s+(?:e\s+)?obbligatorio', "🚶 Sopralluogo obbligatorio"),
            (r'inversione\s+procedimentale', "🔄 Inversione procedimentale (art. 107 c.3)"),
            (r'soglia\s+(?:di\s+)?sbarramento', "⛔ Soglia di sbarramento tecnica"),
            (r'giovani?\s+professionisti?', "👨‍🎓 Bonus giovani professionisti"),
            (r'parita\s+di\s+genere|certificazione\s+parita', "♀♂ Certificazione parità di genere"),
            (r'ISO\s*9001', "📋 Bonus ISO 9001"),
            (r'vincolo.{0,30}(?:entrambi|tutti)\s+(?:i\s+)?lotti', "🔗 Partecipazione obbligatoria tutti i lotti"),
            (r'accordo\s+quadro.{0,30}unico\s+operatore', "📑 Accordo Quadro a unico operatore"),
            (r'conformita.{0,10}CAM|criteri\s+ambientali\s+minimi', "🌿 Conformità CAM obbligatoria"),
            (r'(?:FESR|PNRR|PON|POR)', "🇪🇺 Finanziamento pubblico/europeo"),
            (r'soccorso\s+istruttorio', "📋 Soccorso istruttorio previsto"),
        ]
        for pattern, nota in checks:
            if re.search(pattern, text, re.I):
                note.append(nota)
        return note

    def detect_finanziamento(self, text: str) -> Optional[str]:
        fonti = {
            "PNRR": r'PNRR|Piano\s+Nazionale\s+di\s+Ripresa',
            "FESR": r'FESR|Fondo\s+Europeo\s+di\s+Sviluppo\s+Regionale',
            "PON/POR": r'(?:PON|POR|PR)\s+[A-Za-z\s]+20\d{2}',
            "Decreto Dirigenziale": r'decreto\s+(?:dirigenziale|ministeriale)[^\n]{0,60}',
            "Fondi propri": r'(?:bilancio|fondi)\s+(?:propri|comunali|regionali)',
        }
        for nome, pat in fonti.items():
            if re.search(pat, text, re.I):
                return nome
        return None

    # ── ESTRAZIONE COMPLETA ────────────────────────────────────────

    def extract(self, text: str) -> Tuple[dict, dict, dict]:
        """Estrazione deterministica. Ritorna (result, snippets, methods)."""
        r = {}
        pats = self._patterns

        # Identificativi
        r["cig"] = self.first_match(text, pats.get("cig", []))
        cups = self.all_matches(text, pats.get("cup", []))
        r["cup"] = cups[0] if len(cups) == 1 else (cups if cups else None)
        cpvs = self.all_matches(text, pats.get("cpv", []))
        r["cpv"] = cpvs[0] if len(cpvs) == 1 else (cpvs if cpvs else None)
        r["nuts_code"] = self.first_match(text, pats.get("nuts_code", []))
        r["codice_progetto"] = self.first_match(text, pats.get("codice_progetto", []))

        # Soggetti
        r["stazione_appaltante"] = self.first_match(text, pats.get("stazione_appaltante", []))
        r["amministrazione_delegante"] = self.first_match(text, pats.get("amministrazione_delegante", []))
        r["rup"] = self.first_match(text, pats.get("rup", []))
        r["responsabile_procedimento"] = self.first_match(text, pats.get("responsabile_procedimento", []))
        r["direttore_esecuzione"] = self.first_match(text, pats.get("direttore_esecuzione", []))
        r["coordinatore_sicurezza"] = self.first_match(text, pats.get("coordinatore_sicurezza", []))

        # Oggetto
        r["oggetto_appalto"] = self.first_match(text, pats.get("oggetto_appalto", []))

        # Accordo quadro
        r["is_accordo_quadro"] = bool(re.search(r'accordo\s+quadro', text, re.I))
        if r["is_accordo_quadro"]:
            m = re.search(r'accordo\s+quadro\s+(?:a|con)\s+(unico|doppio)', text, re.I)
            r["tipo_accordo_quadro"] = self.clean(m.group(1)) if m else None

        # Lotti
        n_raw = self.first_match(text, pats.get("numero_lotti", [r'(?:suddivisa?\s+in|articolat[ao]\s+in)\s+(\d+|due|tre|quattro|cinque)\s+lott']))
        r["numero_lotti"] = self.parse_number_word(n_raw)
        r["tipo_gara"] = "MULTILOTTO" if r["numero_lotti"] and r["numero_lotti"] > 1 else "MONOLOTTO"
        r["lotti"] = self.extract_lotti_detail(text)
        r["vincoli_lotti"] = self.extract_vincoli_lotti(text)

        # Importi
        r["importo_totale"] = self.normalize_amount(self.first_match(text, pats.get("importo_totale", [])))
        r["importo_base_gara"] = self.normalize_amount(self.first_match(text, pats.get("importo_base_gara", [])))
        r["oneri_sicurezza"] = self.normalize_amount(self.first_match(text, pats.get("oneri_sicurezza", [])))
        r["costi_manodopera"] = self.normalize_amount(self.first_match(text, pats.get("costi_manodopera", [])))
        r["struttura_compenso_65_35"] = self.extract_struttura_compenso(text)
        r["garanzia_provvisoria"] = self.normalize_amount(self.first_match(text, pats.get("garanzia_provvisoria", [])))
        r["garanzia_definitiva"] = self.first_match(text, pats.get("garanzia_definitiva", []))
        r["anticipazione"] = self.first_match(text, pats.get("anticipazione", []))

        # Punteggi
        r["punteggio_tecnica"] = self.extract_int(text, pats.get("punteggio_tecnica", []))
        r["punteggio_economica"] = self.extract_int(text, pats.get("punteggio_economica", []))
        r["soglia_sbarramento_tecnica"] = self.extract_int(text, pats.get("soglia_sbarramento_tecnica", []))
        r["criteri_tecnici"] = self.extract_criteri_tecnici(text)

        # Bonus tabellari
        r["pti_giovani_professionisti"] = self.extract_int(text, pats.get("pti_giovani_professionisti", []))
        r["pti_iso_9001"] = self.extract_int(text, pats.get("pti_iso_9001", []))
        r["pti_parita_genere"] = self.extract_int(text, pats.get("pti_parita_genere", []))

        # Tempistiche
        r["scadenze"] = self.extract_scadenze(text)
        r["scadenza_offerte"] = self.first_match(text, pats.get("scadenza_offerte", []))
        r["termine_chiarimenti"] = self.first_match(text, pats.get("termine_chiarimenti", []))
        r["durata_contratto"] = self.first_match(text, pats.get("durata_contratto", []))
        r["termine_esecuzione"] = self.first_match(text, pats.get("termine_esecuzione", []))

        # Categorie, requisiti
        r["categorie_ingegneria"] = self.extract_categorie_ingegneria(text)
        r["categorie_gara"] = self.extract_categorie_gara(text)
        r["periodo_requisiti_anni"] = self.extract_int(text, pats.get("periodo_requisiti_anni", []))
        r["fatturato_minimo"] = self.normalize_amount(self.first_match(text, pats.get("fatturato_minimo", [])))
        r["polizza_professionale"] = self.first_match(text, pats.get("polizza_professionale", []))
        r["requisiti_soa"] = self.first_match(text, pats.get("requisiti_soa", []))

        # Figure professionali
        r["cumuli_ammessi"] = bool(re.search(
            r'cumulo|cumulabil[ei]|ruoli?\s+(?:possono|puo)\s+essere\s+cumulat[io]|'
            r'(?:ammess[oa]|consentit[oa])\s+(?:il\s+)?cumulo',
            text, re.I
        ))
        r["richiesto_giovane_professionista"] = bool(re.search(
            r'giovane\s+professionista|professionista\s+(?:abilitato\s+)?da\s+meno\s+di\s+\d+\s+ann[io]|'
            r'art\.?\s*(?:4|5)[^\n]{0,30}giovane',
            text, re.I
        ))
        r["figure_professionali_dettaglio"] = self.extract_figure_professionali(text)

        # Sopralluogo
        sop = self.extract_sopralluogo(text)
        r["sopralluogo_obbligatorio"] = sop.get("obbligatorio", False)
        r["sopralluogo_note"] = sop.get("note")

        # Piattaforma, contributi
        r["piattaforma_url"] = self.first_match(text, pats.get("piattaforma_url", []))
        r["contributo_anac"] = self.normalize_amount(self.first_match(text, pats.get("contributo_anac", [])))
        r["imposta_bollo"] = self.normalize_amount(self.first_match(text, pats.get("imposta_bollo", [])))
        r["subappalto"] = self.first_match(text, pats.get("subappalto", []))
        r["avvalimento"] = self.first_match(text, pats.get("avvalimento", []))

        # Flags booleani
        r["revisione_prezzi"] = bool(re.search(r'revisione\s+(?:dei\s+)?prezzi', text, re.I))
        r["conformita_cam"] = bool(re.search(r'\bCAM\b|criteri\s+ambientali\s+minimi', text, re.I))
        r["inversione_procedimentale"] = bool(re.search(r'inversione\s+procedimentale', text, re.I))
        r["soccorso_istruttorio"] = bool(re.search(r'soccorso\s+istruttorio', text, re.I))
        r["clausola_sociale"] = bool(re.search(r'clausola\s+sociale|riassorbimento\s+(?:del\s+)?personale', text, re.I))

        # Condizioni
        r["penali"] = self.first_match(text, pats.get("penali", []))
        r["modalita_pagamento"] = self.first_match(text, pats.get("modalita_pagamento", []))
        r["garanzia_provvisoria_ridotta"] = self.first_match(text, pats.get("garanzia_provvisoria_ridotta", []))
        r["verifica_anomalia"] = self.first_match(text, pats.get("verifica_anomalia", []))
        r["finanziamento"] = self.detect_finanziamento(text)
        r["note_operative"] = self.extract_note_operative(text)

        # Campi custom
        for fd in registry.get_custom_fields():
            if fd.key not in r and fd.patterns:
                if fd.field_type == "boolean":
                    r[fd.key] = any(re.search(p, text, re.I) for p in fd.patterns)
                elif fd.field_type == "money":
                    r[fd.key] = self.normalize_amount(self.first_match(text, fd.patterns))
                elif fd.field_type == "number":
                    val = self.first_match(text, fd.patterns)
                    try:
                        r[fd.key] = int(re.search(r'\d+', str(val)).group()) if val else None
                    except Exception:
                        r[fd.key] = None
                else:
                    r[fd.key] = self.first_match(text, fd.patterns)

        # Build snippets & methods
        snippets = {}
        methods = {}
        empty_objects = [
            {"quota_fissa_65_perc": False, "quota_ribassabile_35_perc": False},
            {"obbligatorio": False, "note": None},
            {"partecipazione_tutti_lotti": False, "offerta_identica": False,
             "max_lotti_aggiudicazione": None, "medesima_forma_giuridica": False}
        ]
        for key, value in r.items():
            if key.startswith('_') or value is None:
                continue
            if isinstance(value, bool):
                if value:
                    methods[key] = "rules"
                continue
            if isinstance(value, dict):
                if any(v for v in value.values() if v not in [None, False, ""]):
                    methods[key] = "rules"
                continue
            if isinstance(value, list):
                if len(value) > 0:
                    methods[key] = "rules"
                    ctx = find_value_context(text, str(value[0]) if value else "")
                    if ctx:
                        snippets[key] = ctx
                continue
            if value in ["", 0]:
                continue
            methods[key] = "rules"
            ctx = find_value_context(text, str(value))
            if ctx:
                snippets[key] = ctx

        return r, snippets, methods
