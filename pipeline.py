"""
AppaltoAI — Pipeline di Estrazione a 9 Fasi (SELF-CONTAINED)
═════════════════════════════════════════════════════════════
Architettura completa con separazione rigorosa delle responsabilità:

  Fase 1: Upload + Hash Dedup
  Fase 2: Parsing deterministico (solo regole, nessuna AI)
  Fase 3: NLP specializzato (classificazione, entity recognition — NO generazione testo)
  Fase 4: Costruzione JSON finale (merge regole + NLP — responsabilità del CODICE)
  Fase 5: Validazione automatica (schema Pydantic, somme, coerenza)
  Fase 6: Output strutturato + salvataggio
  Fase 7: Gestione correzioni → dataset annotato proprietario
  Fase 8: Retraining controllato (SOLO supervisionato, MAI automatico)
  Fase 9: Versionamento modelli (v1, v2, v3 con rollback)

PRINCIPI CHIAVE:
  - Il modello NON deve scrivere testo o fare interpretazioni creative
  - Il modello: riconosce entità, classifica, struttura
  - JSON è responsabilità del CODICE, non del modello
  - Mai training automatico. Sempre supervisionato dall'admin.
"""

import hashlib
import io
import json
import pickle
import re
import shutil
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

from field_registry import registry, get_validator
from pdf_parser import parse_pdf, get_text_with_tables, get_page_for_text, ParsedDocument
from schemas import full_validation

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DB_PATH = DATA_DIR / "learning.db"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# FASE 1: UPLOAD + HASH DEDUP
# ═════════════════════════════════════════════════════════════════════════════

def compute_hash(content: bytes | str) -> str:
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def check_duplicate(text_hash: str) -> Optional[dict]:
    try:
        conn = sqlite3.connect(str(DB_PATH))
        row = conn.execute(
            "SELECT id, filename, extracted_json, corrected_json "
            "FROM documents WHERE text_hash=? ORDER BY upload_date DESC LIMIT 1",
            (text_hash,)
        ).fetchone()
        conn.close()
        if row and row[2]:
            result = json.loads(row[2])
            result["_cached"] = True
            result["_cached_doc_id"] = row[0]
            result["_cached_filename"] = row[1]
            if row[3]:
                corrected = json.loads(row[3])
                for k, v in corrected.items():
                    if v is not None:
                        result[k] = v
                        result.setdefault("_methods", {})[k] = "corrected"
            return result
    except Exception:
        pass
    return None


# ═════════════════════════════════════════════════════════════════════════════
# FASE 2: ESTRAZIONE DETERMINISTICA (SOLO REGOLE — NO AI)
# ═════════════════════════════════════════════════════════════════════════════

class RulesExtractor:
    """Estrazione puramente deterministica basata su regex.
    Nessun modello ML coinvolto in questa fase."""

    def __init__(self):
        self._patterns = registry.get_patterns()

    def reload_patterns(self):
        self._patterns = registry.get_patterns()

    # ── Utility ────────────────────────────────────────────────────

    @staticmethod
    def clean(s: str) -> str:
        if not s:
            return s
        s = re.sub(r'\s+', ' ', s).strip()
        s = re.sub(r'[,;:\s]+$', '', s)
        return s

    @staticmethod
    def normalize_amount(raw) -> Optional[str]:
        if not raw:
            return None
        raw = str(raw).strip().replace("EUR", "").replace("€", "").strip()
        if re.search(r'\d{1,3}(?:\.\d{3})+,\d{2}', raw):
            raw = raw.replace(".", "").replace(",", ".")
        elif re.search(r'\d{1,3}(?:,\d{3})+\.\d{2}', raw):
            raw = raw.replace(",", "")
        else:
            raw = raw.replace(",", ".")
        try:
            num = float(re.sub(r'[^\d.]', '', raw))
            return f"€ {num:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            return raw

    def first_match(self, text: str, patterns: list) -> Optional[str]:
        for pat in patterns:
            try:
                m = re.search(pat, text, re.I | re.M)
                if m:
                    return self.clean(m.group(1) if m.lastindex else m.group(0))
            except Exception:
                continue
        return None

    def all_matches(self, text: str, patterns: list) -> list:
        found = []
        for pat in patterns:
            try:
                found.extend(re.findall(pat, text, re.I | re.M))
            except Exception:
                continue
        return list(dict.fromkeys([self.clean(x) for x in found if x]))

    def extract_int(self, text: str, patterns: list) -> Optional[int]:
        val = self.first_match(text, patterns)
        if val:
            try:
                return int(re.sub(r'\D', '', val))
            except Exception:
                return None
        return None

    def parse_number_word(self, s) -> Optional[int]:
        if not s:
            return None
        mapping = {"due": 2, "tre": 3, "quattro": 4, "cinque": 5, "sei": 6}
        sl = s.strip().lower()
        if sl in mapping:
            return mapping[sl]
        try:
            return int(re.sub(r'\D', '', s))
        except Exception:
            return None

    # ── Estrazioni complesse ───────────────────────────────────────

    def extract_lotti_detail(self, text: str) -> list:
        lotti = []
        parts = re.split(r'(?i)\bLotto\s+(\d+)\b', text)
        if len(parts) > 2:
            for i in range(1, len(parts) - 1, 2):
                n = parts[i]; body = parts[i + 1][:2000]
                cig_m = re.search(r'\bCIG[:\s]*([A-Z0-9]{10})\b', body, re.I)
                imp_m = re.search(r'(?:importo|base\s+gara)[:\s€]*([0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})?)', body, re.I)
                desc_m = re.search(r'^(.{10,200})', body.strip())
                lotti.append({
                    "numero": int(n),
                    "cig": cig_m.group(1) if cig_m else None,
                    "importo": self.normalize_amount(imp_m.group(1)) if imp_m else None,
                    "descrizione": self.clean(desc_m.group(1))[:200] if desc_m else None,
                })
        return lotti

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
        r["periodo_requisiti_anni"] = self.extract_int(text, pats.get("periodo_requisiti_anni", []))
        r["fatturato_minimo"] = self.normalize_amount(self.first_match(text, pats.get("fatturato_minimo", [])))
        r["polizza_professionale"] = self.first_match(text, pats.get("polizza_professionale", []))
        r["requisiti_soa"] = self.first_match(text, pats.get("requisiti_soa", []))

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
        for key, value in r.items():
            if key.startswith('_') or value is None:
                continue
            if isinstance(value, bool):
                if value: methods[key] = "rules"
                continue
            if isinstance(value, dict):
                if any(v for v in value.values() if v not in [None, False, ""]):
                    methods[key] = "rules"
                continue
            if isinstance(value, list):
                if len(value) > 0:
                    methods[key] = "rules"
                    ctx = _find_value_context(text, str(value[0]) if value else "")
                    if ctx: snippets[key] = ctx
                continue
            if value in ["", 0]:
                continue
            methods[key] = "rules"
            ctx = _find_value_context(text, str(value))
            if ctx: snippets[key] = ctx

        return r, snippets, methods


# ═════════════════════════════════════════════════════════════════════════════
# FASE 3: NLP SPECIALIZZATO (solo classificazione + entity recognition)
# ═════════════════════════════════════════════════════════════════════════════

class NLPClassifier:
    """Classificatori ML specializzati. Il modello NON genera testo."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self._load_models()

    def _load_models(self):
        for mf in MODEL_DIR.glob("model_*.pkl"):
            field = mf.stem.replace("model_", "")
            try:
                with open(mf, "rb") as f:
                    self.models[field] = pickle.load(f)
            except Exception:
                pass

    def reload(self):
        self.models.clear()
        self._load_models()

    def classify_procedure(self, text: str) -> str:
        if re.search(r'accordo\s+quadro', text, re.I):
            tipo = "a unico operatore" if re.search(r'unico\s+operatore', text, re.I) else ""
            n = re.search(r'suddivisa?\s+in\s+(\d+|due|tre)\s+lott', text, re.I)
            lotti = f" — {n.group(1)} Lotti" if n else ""
            return f"Procedura Aperta Telematica — Accordo Quadro {tipo}{lotti}".strip()
        if re.search(r'procedura\s+(?:telematica\s+)?aperta', text, re.I):
            n = re.search(r'suddivisa?\s+in\s+(\d+|due|tre)\s+lott', text, re.I)
            lotti = f" — {n.group(1)} Lotti" if n else ""
            return f"Procedura Aperta Telematica{lotti}"
        if re.search(r'procedura\s+ristretta', text, re.I):
            return "Procedura Ristretta"
        if re.search(r'procedura\s+negoziata', text, re.I):
            return "Procedura Negoziata"
        if re.search(r'affidamento\s+diretto', text, re.I):
            return "Affidamento Diretto"
        if re.search(r'(?:RdO|MEPA|mercato\s+elettronico)', text, re.I):
            return "RdO / MePA"
        pred = self._predict("tipo_procedura", text[:3000])
        return pred if pred else "Non specificata"

    def classify_criterio(self, text: str) -> str:
        if re.search(r'offerta\s+economicamente\s+piu\s+vantaggiosa|OEPV', text, re.I):
            art = re.search(r'art(?:icolo)?\.?\s*108\s+comma\s+\d+', text, re.I)
            suffix = f" ({RulesExtractor.clean(art.group(0))} D.Lgs. 36/2023)" if art else " (art. 108 D.Lgs. 36/2023)"
            return "OEPV — Offerta Economicamente Più Vantaggiosa" + suffix
        if re.search(r'(?:massimo|minor|minimo)\s+ribasso', text, re.I):
            return "Massimo/Minor Ribasso"
        if re.search(r'prezzo\s+piu\s+basso', text, re.I):
            return "Prezzo più basso"
        pred = self._predict("criterio_aggiudicazione", text[:3000])
        return pred if pred else "Non specificato"

    def fill_missing(self, result: dict, text: str) -> Tuple[dict, dict]:
        """ML fallback per campi vuoti. Ritorna (result, ml_methods)."""
        ml_methods = {}
        for key, value in result.items():
            if key.startswith('_') or isinstance(value, (bool, dict, list)):
                continue
            if value in [None, "", 0] and key in self.models:
                pred = self._predict(key, text[:3000])
                if pred:
                    result[key] = pred
                    ml_methods[key] = "ml"
        return result, ml_methods

    def _predict(self, field: str, snippet: str) -> Optional[str]:
        model = self.models.get(field)
        if model and snippet:
            try:
                return model.predict([snippet])[0]
            except Exception:
                return None
        return None


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPALE — Orchestratore completo
# ═════════════════════════════════════════════════════════════════════════════

class Pipeline:
    """Orchestratore della pipeline a 9 fasi."""

    def __init__(self):
        self.rules = RulesExtractor()
        self.nlp = NLPClassifier()
        self._last_parsed: Dict[str, ParsedDocument] = {}
        self._init_db()

    def _init_db(self):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT,
                text_hash TEXT,
                full_text TEXT,
                upload_date TEXT,
                extracted_json TEXT,
                corrected_json TEXT,
                model_version TEXT,
                score REAL DEFAULT 0,
                feedback TEXT
            );
            CREATE TABLE IF NOT EXISTS training_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                field TEXT NOT NULL,
                text_snippet TEXT NOT NULL,
                correct_value TEXT NOT NULL,
                wrong_value TEXT,
                source TEXT DEFAULT 'manual',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                field TEXT,
                version INTEGER,
                accuracy REAL,
                samples_count INTEGER,
                trained_at TEXT DEFAULT CURRENT_TIMESTAMP,
                is_active INTEGER DEFAULT 1,
                notes TEXT
            );
            CREATE TABLE IF NOT EXISTS feedback_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                field TEXT,
                original TEXT,
                corrected TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        # Migrazioni sicure
        for col, tbl, ctype in [
            ('full_text', 'documents', 'TEXT'),
            ('model_version', 'documents', 'TEXT'),
            ('source', 'training_samples', 'TEXT DEFAULT "manual"'),
            ('is_active', 'model_versions', 'INTEGER DEFAULT 1'),
            ('notes', 'model_versions', 'TEXT'),
        ]:
            try:
                c.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {ctype}")
            except Exception:
                pass
        conn.commit()
        conn.close()

    # ═══════════════════════════════════════════════════════════════
    # FASI 1-6: Estrazione completa
    # ═══════════════════════════════════════════════════════════════

    def process_pdf(self, pdf_bytes: bytes, filename: str) -> dict:
        """Pipeline completa per PDF."""
        t_start = time.time()

        # FASE 1: Hash dedup
        pdf_hash = compute_hash(pdf_bytes)
        cached = check_duplicate(pdf_hash)
        if cached:
            cached["_pipeline_phase"] = "cached"
            cached["_pipeline_time_ms"] = round((time.time() - t_start) * 1000, 1)
            return cached

        # FASE 2: Parse PDF
        text, parsed = self._parse_pdf(pdf_bytes, filename)
        if not text or len(text.strip()) < 50:
            return {"error": "Impossibile estrarre testo dal PDF", "_pipeline_phase": "parse_failed"}

        # FASI 3-6
        result = self._extract_and_build(text, filename)

        # Aggiungi metadati PDF
        if parsed and parsed.pages:
            page_sources = {}
            for field, value in result.items():
                if field.startswith('_') or value is None or value == "" or isinstance(value, (bool, dict, list)):
                    continue
                page = get_page_for_text(parsed, str(value))
                if page:
                    page_sources[field] = page
            result["_page_sources"] = page_sources
            result["_total_pages"] = parsed.total_pages
            result["_is_native_pdf"] = parsed.is_native
            result["_parser_used"] = parsed.parser_used
            result["_pdf_metadata"] = parsed.metadata
            result["_pdf_warnings"] = parsed.warnings
            result["_tables_found"] = len(parsed.tables_json)
            result["_tables_json"] = parsed.tables_json[:20]
            result["_chunks_count"] = len(parsed.chunks)

        # Salva con hash
        self._save_document(result["_doc_id"], filename, text, result, pdf_hash)
        result["_pipeline_phase"] = "complete"
        result["_pipeline_time_ms"] = round((time.time() - t_start) * 1000, 1)
        return result

    def process_text(self, text: str, filename: str = "input.txt") -> dict:
        """Pipeline per testo grezzo."""
        t_start = time.time()

        # Hash dedup su testo
        text_hash = compute_hash(text)
        cached = check_duplicate(text_hash)
        if cached:
            cached["_pipeline_phase"] = "cached"
            return cached

        result = self._extract_and_build(text, filename)
        self._save_document(result["_doc_id"], filename, text, result, text_hash)
        result["_pipeline_phase"] = "complete"
        result["_pipeline_time_ms"] = round((time.time() - t_start) * 1000, 1)
        return result

    def _extract_and_build(self, text: str, filename: str) -> dict:
        """Fasi 3-6: Rules → NLP → Build JSON → Validate."""
        # FASE 2: Estrazione deterministica
        rules_result, snippets, methods = self.rules.extract(text)

        # FASE 3: NLP classificazione
        rules_result["tipo_procedura"] = self.nlp.classify_procedure(text)
        rules_result["criterio_aggiudicazione"] = self.nlp.classify_criterio(text)
        methods["tipo_procedura"] = "rules"
        methods["criterio_aggiudicazione"] = "rules"

        # FASE 3b: ML fallback campi vuoti
        rules_result, ml_methods = self.nlp.fill_missing(rules_result, text)
        methods.update(ml_methods)
        for key in ml_methods:
            if key not in snippets:
                ctx = _find_value_context(text, str(rules_result.get(key, "")))
                if ctx:
                    snippets[key] = ctx

        # FASE 4: Costruzione JSON (responsabilità del CODICE)
        doc_id = hashlib.sha256((filename + text[:500]).encode()).hexdigest()[:16]
        result = dict(rules_result)
        result["_snippets"] = snippets
        result["_methods"] = methods
        result["_doc_id"] = doc_id
        result["_model_version"] = self._get_active_model_version()

        # Confidence
        empty_objects = [
            {"quota_fissa_65_perc": False, "quota_ribassabile_35_perc": False},
            {"obbligatorio": False, "note": None},
            {"partecipazione_tutti_lotti": False, "offerta_identica": False,
             "max_lotti_aggiudicazione": None, "medesima_forma_giuridica": False}
        ]
        total = sum(1 for k in result if not k.startswith('_'))
        filled = sum(1 for k, v in result.items()
                     if not k.startswith('_') and v not in [None, [], {}, False, "", 0]
                     and v not in empty_objects)
        result["_confidence"] = round(filled / max(total, 1) * 100, 1)
        result["_extraction_method"] = "hybrid" if ml_methods else "rules"
        result["_timestamp"] = datetime.now().isoformat()
        result["_text_length"] = len(text)
        result["_filename"] = filename

        # FASE 5: Validazione
        try:
            validation = full_validation(result)
            result["_coherence"] = validation["coherence"]
            result["_validation_warnings"] = validation["warnings"]
        except Exception:
            pass

        return result

    def _parse_pdf(self, pdf_bytes: bytes, filename: str) -> Tuple[str, Optional[ParsedDocument]]:
        parsed = None
        try:
            parsed = parse_pdf(pdf_bytes, filename)
            self._last_parsed[filename] = parsed
            text = get_text_with_tables(parsed)
            if text and len(text.strip()) > 50:
                return text, parsed
        except Exception:
            pass
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
            if text.strip():
                return text, parsed
        except Exception:
            pass
        return "", None

    def get_last_parsed(self, filename: str) -> Optional[ParsedDocument]:
        return self._last_parsed.get(filename)

    # ═══════════════════════════════════════════════════════════════
    # FASE 7: CORREZIONI → DATASET ANNOTATO PROPRIETARIO
    # ═══════════════════════════════════════════════════════════════

    def record_correction(self, doc_id: str, field: str, original: str,
                          corrected: str, snippet: str = "") -> dict:
        """Registra correzione → annotated training example.
        NON attiva auto-training. Solo salva nel dataset."""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        row_text = c.execute("SELECT full_text FROM documents WHERE id=?", (doc_id,)).fetchone()
        full_text = row_text[0] if row_text and row_text[0] else ""
        training_snippet = snippet
        if full_text:
            ctx = _find_value_context(full_text, corrected, window=500)
            if ctx and len(ctx) > 20:
                training_snippet = ctx
            elif original:
                ctx = _find_value_context(full_text, original, window=500)
                if ctx and len(ctx) > 20:
                    training_snippet = ctx
        if not training_snippet:
            training_snippet = full_text[:1500]
        c.execute("INSERT INTO feedback_log (doc_id, field, original, corrected) VALUES (?,?,?,?)",
                  (doc_id, field, original, corrected))
        c.execute("INSERT INTO training_samples (field, text_snippet, correct_value, wrong_value, source) VALUES (?,?,?,?,?)",
                  (field, training_snippet[:2000], corrected, original, "correction"))
        row = c.execute("SELECT corrected_json FROM documents WHERE id=?", (doc_id,)).fetchone()
        cd = json.loads(row[0]) if row and row[0] else {}
        cd[field] = corrected
        c.execute("UPDATE documents SET corrected_json=? WHERE id=?",
                  (json.dumps(cd, ensure_ascii=False), doc_id))
        conn.commit()
        conn.close()
        count = self._get_sample_count(field)
        return {"status": "ok", "field": field, "sample_count": count,
                "message": f"Correzione salvata ({count} campioni). Training manuale disponibile."}

    # ═══════════════════════════════════════════════════════════════
    # FASE 8: RETRAINING SUPERVISIONATO (MAI AUTOMATICO)
    # ═══════════════════════════════════════════════════════════════

    def train_field(self, field: str) -> dict:
        """Retraining supervisionato. Solo l'admin lo attiva."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import Pipeline as SkPipeline
        import random

        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        rows = c.execute("SELECT text_snippet, correct_value FROM training_samples WHERE field=?", (field,)).fetchall()
        if len(rows) < 3:
            conn.close()
            return {"status": "error", "message": f"Campioni insufficienti ({len(rows)}/3) per '{field}'."}

        texts = [r[0] for r in rows]
        labels = [r[1] for r in rows]
        old_model = self.nlp.models.get(field)
        old_accuracy = new_accuracy = None

        if len(rows) >= 6:
            indices = list(range(len(rows)))
            random.seed(42); random.shuffle(indices)
            split = max(2, int(len(indices) * 0.2))
            test_idx = set(indices[:split])
            train_texts = [texts[i] for i in indices if i not in test_idx]
            train_labels = [labels[i] for i in indices if i not in test_idx]
            test_texts = [texts[i] for i in test_idx]
            test_labels = [labels[i] for i in test_idx]
            if len(set(train_labels)) >= 2 or len(set(labels)) == 1:
                if old_model:
                    try:
                        old_preds = old_model.predict(test_texts)
                        old_accuracy = sum(1 for p, t in zip(old_preds, test_labels) if p == t) / len(test_labels)
                    except Exception:
                        pass
                try:
                    tp = SkPipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=8000, analyzer="char_wb")),
                                     ("clf", SGDClassifier(loss="hinge", max_iter=1000, random_state=42))])
                    tp.fit(train_texts, train_labels)
                    new_preds = tp.predict(test_texts)
                    new_accuracy = sum(1 for p, t in zip(new_preds, test_labels) if p == t) / len(test_labels)
                except Exception:
                    pass
                if old_accuracy is not None and new_accuracy is not None and new_accuracy < old_accuracy - 0.05:
                    conn.close()
                    return {"status": "rollback",
                            "message": f"⚠ Nuovo modello peggio ({new_accuracy:.0%} vs {old_accuracy:.0%}). Mantenuto vecchio.",
                            "old_accuracy": old_accuracy, "new_accuracy": new_accuracy}

        pipeline_model = SkPipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=8000, analyzer="char_wb")),
                                     ("clf", SGDClassifier(loss="hinge", max_iter=1000, random_state=42))])
        pipeline_model.fit(texts, labels)

        path = MODEL_DIR / f"model_{field}.pkl"
        backup_path = MODEL_DIR / f"model_{field}_prev.pkl"
        if path.exists():
            shutil.copy2(path, backup_path)
        with open(path, "wb") as f:
            pickle.dump(pipeline_model, f)
        self.nlp.models[field] = pipeline_model

        # FASE 9: Versionamento
        current_v = c.execute("SELECT MAX(version) FROM model_versions WHERE field=?", (field,)).fetchone()[0] or 0
        new_v = current_v + 1
        c.execute("UPDATE model_versions SET is_active=0 WHERE field=?", (field,))
        c.execute("INSERT INTO model_versions (field, version, accuracy, samples_count, trained_at, is_active, notes) VALUES (?,?,?,?,?,1,?)",
                  (field, new_v, new_accuracy, len(rows), datetime.now().isoformat(),
                   f"Supervised. Old: {old_accuracy}, New: {new_accuracy}"))
        conn.commit()
        conn.close()

        msg = f"Modello '{field}' v{new_v} — {len(rows)} campioni"
        if new_accuracy: msg += f" (acc: {new_accuracy:.0%})"
        if old_accuracy: msg += f" [prima: {old_accuracy:.0%}]"
        return {"status": "ok", "message": msg, "field": field, "version": new_v,
                "accuracy": new_accuracy, "old_accuracy": old_accuracy, "samples": len(rows)}

    def rollback_model(self, field: str) -> dict:
        """Rollback al modello precedente."""
        backup = MODEL_DIR / f"model_{field}_prev.pkl"
        path = MODEL_DIR / f"model_{field}.pkl"
        if not backup.exists():
            return {"status": "error", "message": f"Nessun backup per '{field}'"}
        shutil.copy2(backup, path)
        with open(path, "rb") as f:
            self.nlp.models[field] = pickle.load(f)
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("UPDATE model_versions SET is_active=0 WHERE field=?", (field,))
        c.execute("UPDATE model_versions SET is_active=1 WHERE field=? AND id=(SELECT MAX(id)-1 FROM model_versions WHERE field=?)", (field, field))
        conn.commit()
        conn.close()
        return {"status": "ok", "message": f"Rollback '{field}' completato"}

    # ═══════════════════════════════════════════════════════════════
    # FASE 9: MODEL VERSIONING
    # ═══════════════════════════════════════════════════════════════

    def get_model_versions(self, field: str = None) -> list:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        if field:
            rows = c.execute("SELECT field, version, accuracy, samples_count, trained_at, is_active, notes FROM model_versions WHERE field=? ORDER BY version DESC", (field,)).fetchall()
        else:
            rows = c.execute("SELECT field, version, accuracy, samples_count, trained_at, is_active, notes FROM model_versions ORDER BY trained_at DESC").fetchall()
        conn.close()
        return [{"field": r[0], "version": r[1], "accuracy": r[2], "samples": r[3],
                 "trained_at": r[4], "is_active": bool(r[5]), "notes": r[6]} for r in rows]

    def _get_active_model_version(self) -> str:
        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute("SELECT field, version FROM model_versions WHERE is_active=1").fetchall()
        conn.close()
        if not rows: return "rules-only"
        return ", ".join(f"{r[0]}:v{r[1]}" for r in rows)

    # ═══════════════════════════════════════════════════════════════
    # CORRECTIONS CRUD
    # ═══════════════════════════════════════════════════════════════

    def get_corrections(self, limit: int = 200) -> list:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        rows = c.execute("""
            SELECT f.id, f.doc_id, f.field, f.original, f.corrected, f.timestamp,
                   t.id as sample_id, t.text_snippet
            FROM feedback_log f
            LEFT JOIN training_samples t ON t.field = f.field AND t.correct_value = f.corrected AND t.wrong_value = f.original
            ORDER BY f.timestamp DESC LIMIT ?
        """, (limit,)).fetchall()
        corrections = []
        seen = set()
        for r in rows:
            key = (r[0], r[6])
            if key in seen: continue
            seen.add(key)
            corrections.append({"id": r[0], "doc_id": r[1], "field": r[2], "original": r[3],
                                "corrected": r[4], "timestamp": r[5], "sample_id": r[6], "snippet": (r[7] or "")[:300]})
        manual = c.execute("""
            SELECT t.id, t.field, t.text_snippet, t.correct_value, t.wrong_value, t.created_at, t.source
            FROM training_samples t WHERE NOT EXISTS (SELECT 1 FROM feedback_log f WHERE f.field=t.field AND f.corrected=t.correct_value)
            ORDER BY t.created_at DESC LIMIT ?
        """, (limit,)).fetchall()
        for r in manual:
            corrections.append({"id": None, "doc_id": None, "field": r[1], "original": r[4] or "",
                                "corrected": r[3], "timestamp": r[5], "sample_id": r[0],
                                "snippet": (r[2] or "")[:300], "source": r[6] or "training_sample"})
        conn.close()
        corrections.sort(key=lambda x: x.get("timestamp") or "", reverse=True)
        return corrections[:limit]

    def update_correction(self, correction_id=None, sample_id=None, data=None) -> dict:
        if not data: return {"status": "error", "message": "Nessun dato"}
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor(); updated = 0
        nc = data.get("corrected"); nf = data.get("field"); ns = data.get("snippet")
        if correction_id:
            if nc: c.execute("UPDATE feedback_log SET corrected=? WHERE id=?", (nc, correction_id)); updated += 1
            if nf: c.execute("UPDATE feedback_log SET field=? WHERE id=?", (nf, correction_id))
        if sample_id:
            if nc: c.execute("UPDATE training_samples SET correct_value=? WHERE id=?", (nc, sample_id)); updated += 1
            if ns: c.execute("UPDATE training_samples SET text_snippet=? WHERE id=?", (ns[:2000], sample_id))
            if nf: c.execute("UPDATE training_samples SET field=? WHERE id=?", (nf, sample_id))
        conn.commit(); conn.close()
        return {"status": "ok", "updated": updated}

    def delete_correction(self, correction_id=None, sample_id=None) -> dict:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor(); deleted = 0
        if correction_id:
            row = c.execute("SELECT field, corrected, original FROM feedback_log WHERE id=?", (correction_id,)).fetchone()
            if row:
                c.execute("DELETE FROM feedback_log WHERE id=?", (correction_id,))
                c.execute("DELETE FROM training_samples WHERE field=? AND correct_value=? AND (wrong_value=? OR wrong_value IS NULL)", row)
                deleted += 1
        if sample_id:
            c.execute("DELETE FROM training_samples WHERE id=?", (sample_id,)); deleted += 1
        conn.commit(); conn.close()
        return {"status": "ok", "deleted": deleted}

    # ═══════════════════════════════════════════════════════════════
    # STATS & HISTORY
    # ═══════════════════════════════════════════════════════════════

    def get_stats(self) -> dict:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        docs = c.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        corrections = c.execute("SELECT COUNT(*) FROM feedback_log").fetchone()[0]
        samples = c.execute("SELECT field, COUNT(*) FROM training_samples GROUP BY field").fetchall()
        models = c.execute("SELECT field, version, accuracy, samples_count, trained_at, is_active FROM model_versions ORDER BY trained_at DESC").fetchall()
        conn.close()
        return {
            "total_documents": docs,
            "total_corrections": corrections,
            "training_samples": {s[0]: s[1] for s in samples},
            "trained_models": [{"field": m[0], "version": m[1], "accuracy": m[2], "samples": m[3],
                                "trained_at": m[4], "is_active": bool(m[5])} for m in models],
            "loaded_ml_models": list(self.nlp.models.keys()),
        }

    def get_history(self) -> list:
        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute("SELECT id, filename, upload_date, extracted_json, model_version FROM documents ORDER BY upload_date DESC LIMIT 50").fetchall()
        conn.close()
        result = []
        for r in rows:
            try:
                data = json.loads(r[3]) if r[3] else {}
                result.append({"id": r[0], "filename": r[1], "date": r[2],
                               "oggetto": data.get("oggetto_appalto", "N/D"),
                               "importo": data.get("importo_totale") or data.get("importo_base_gara", "N/D"),
                               "stazione": data.get("stazione_appaltante", "N/D"),
                               "confidence": data.get("_confidence", 0),
                               "lotti": data.get("numero_lotti"),
                               "procedura": data.get("tipo_procedura", "N/D"),
                               "model_version": r[4] or "rules-only"})
            except Exception: pass
        return result

    def get_document_text(self, doc_id: str) -> Optional[dict]:
        conn = sqlite3.connect(str(DB_PATH))
        row = conn.execute("SELECT full_text, extracted_json, corrected_json FROM documents WHERE id=?", (doc_id,)).fetchone()
        conn.close()
        if row:
            return {"text": row[0] or "", "extracted": json.loads(row[1]) if row[1] else {}, "corrected": json.loads(row[2]) if row[2] else {}}
        return None

    def get_corrections_stats(self) -> dict:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        stats = c.execute("SELECT field, COUNT(*) FROM feedback_log GROUP BY field ORDER BY COUNT(*) DESC").fetchall()
        total = c.execute("SELECT COUNT(*) FROM feedback_log").fetchone()[0]
        samples = c.execute("SELECT COUNT(*) FROM training_samples").fetchone()[0]
        conn.close()
        return {"total_corrections": total, "total_samples": samples, "by_field": {s[0]: s[1] for s in stats}}

    def _save_document(self, doc_id, filename, text, extracted, text_hash=None):
        conn = sqlite3.connect(str(DB_PATH))
        if not text_hash: text_hash = hashlib.md5(text.encode()).hexdigest()
        conn.execute("INSERT OR REPLACE INTO documents (id, filename, text_hash, full_text, upload_date, extracted_json, model_version) VALUES (?,?,?,?,?,?,?)",
                     (doc_id, filename, text_hash, text, datetime.now().isoformat(),
                      json.dumps(extracted, ensure_ascii=False, default=str), extracted.get("_model_version", "rules-only")))
        conn.commit(); conn.close()

    def _get_sample_count(self, field: str) -> int:
        conn = sqlite3.connect(str(DB_PATH))
        n = conn.execute("SELECT COUNT(*) FROM training_samples WHERE field=?", (field,)).fetchone()[0]
        conn.close()
        return n


# ═════════════════════════════════════════════════════════════════════════════
# UTILITY
# ═════════════════════════════════════════════════════════════════════════════

def _find_value_context(text: str, value: str, window: int = 300) -> str:
    if not value or not text: return ""
    val_str = str(value).strip()
    if len(val_str) < 2: return ""
    idx = text.lower().find(val_str.lower())
    if idx >= 0:
        return text[max(0, idx - window):min(len(text), idx + len(val_str) + window)].strip()
    for word in [w for w in val_str.split() if len(w) > 3]:
        idx = text.lower().find(word.lower())
        if idx >= 0:
            return text[max(0, idx - window):min(len(text), idx + len(word) + window)].strip()
    for num in re.findall(r'\d{3,}', val_str):
        idx = text.find(num)
        if idx >= 0:
            return text[max(0, idx - window):min(len(text), idx + len(num) + window)].strip()
    return ""


# ── Singleton globale ──────────────────────────────────────────────────
pipeline = Pipeline()
