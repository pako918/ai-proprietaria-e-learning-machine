"""
AppaltoAI - Motore Ibrido CALIBRATO
Addestrato su bandi reali D.Lgs. 36/2023 - Servizi di Ingegneria e Architettura
Pattern derivati da:
  - Città Metropolitana di Napoli (SI049/2025) - 3 lotti accordo quadro
  - CUC Valle del Sabato / Comune di Serino (AV) - DL+CSE torrente
  - Risorse per Roma S.p.A. - ERP progettazione esecutiva 2 lotti
"""

import re
import json
import sqlite3
import hashlib
import pickle
from datetime import datetime
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DB_PATH   = DATA_DIR / "learning.db"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# VALIDATORI FORMATO — ogni campo ha una regola di validazione
# Solo i valori che passano la validazione possono entrare nel safe auto-learn
# ═════════════════════════════════════════════════════════════════════════════

def _valid_importo(v):
    """Verifica che sia un importo monetario sensato (> 0 e < 10 miliardi)"""
    try:
        s = str(v).replace('€','').replace('.','').replace(',','.').strip()
        num = float(re.sub(r'[^\d.]', '', s))
        return 0 < num < 10_000_000_000
    except: return False

def _valid_punteggio(v):
    try: return 0 < int(v) <= 100
    except: return False

def _valid_date(v):
    return bool(re.search(r'\d{1,2}[/\-.][\d]{1,2}[/\-.][\d]{2,4}', str(v)))

def _valid_nonempty(v):
    return isinstance(v, str) and len(v.strip()) >= 3

FIELD_VALIDATORS = {
    # Identificativi — formato rigido
    "cig":  lambda v: bool(re.match(r'^[A-Z0-9]{10}$', str(v).strip())),
    "cup":  lambda v: bool(re.match(r'^[A-Z]\d{2}[A-Z]\d{11}$', str(v).strip())),
    "cpv":  lambda v: bool(re.match(r'^\d{8}-\d$', str(v).strip())),
    "nuts_code": lambda v: bool(re.match(r'^IT[A-Z]\d{2,3}$', str(v).strip())),
    # Soggetti — devono avere almeno 5 char
    "stazione_appaltante":       lambda v: isinstance(v, str) and len(v.strip()) >= 5,
    "amministrazione_delegante": lambda v: isinstance(v, str) and len(v.strip()) >= 5,
    "rup":                       lambda v: isinstance(v, str) and len(v.strip()) >= 4,
    "responsabile_procedimento": lambda v: isinstance(v, str) and len(v.strip()) >= 4,
    # Oggetto
    "oggetto_appalto":    lambda v: isinstance(v, str) and len(v.strip()) >= 15,
    "tipo_procedura":     lambda v: isinstance(v, str) and v != "Non specificata" and len(v) > 3,
    "criterio_aggiudicazione": lambda v: isinstance(v, str) and v != "Non specificato" and len(v) > 3,
    # Importi
    "importo_totale":       _valid_importo,
    "importo_base_gara":    _valid_importo,
    "garanzia_provvisoria": _valid_importo,
    "contributo_anac":      _valid_importo,
    "imposta_bollo":        _valid_importo,
    "fatturato_minimo":     _valid_importo,
    # Punteggi (0 < p <= 100)
    "punteggio_tecnica":            _valid_punteggio,
    "punteggio_economica":          _valid_punteggio,
    "soglia_sbarramento_tecnica":   _valid_punteggio,
    "pti_giovani_professionisti":   _valid_punteggio,
    "pti_iso_9001":                 _valid_punteggio,
    "pti_parita_genere":            _valid_punteggio,
    # Date
    "termine_chiarimenti":  _valid_date,
    # Testo libero
    "durata_contratto":     _valid_nonempty,
    "piattaforma_url":      lambda v: isinstance(v, str) and v.startswith('http'),
    "subappalto":           _valid_nonempty,
    "avvalimento":          _valid_nonempty,
    "verifica_anomalia":    _valid_nonempty,
    "finanziamento":        _valid_nonempty,
    "garanzia_definitiva":  _valid_nonempty,
}

# Soglie di confidenza per auto-learn sicuro
AUTO_LEARN_THRESHOLD = 0.85   # >= 85% → auto-learn
QUARANTINE_THRESHOLD = 0.50   # 50-85% → quarantena (review umana)
# Campi ESCLUSI dall'auto-learn (troppo rischiosi o complessi)
AUTO_LEARN_BLACKLIST = {'note_operative', 'lotti', 'criteri_tecnici', 'vincoli_lotti',
                        'scadenze', 'struttura_compenso_65_35', 'sopralluogo',
                        'categorie_ingegneria'}

# ═════════════════════════════════════════════════════════════════════════════
# PATTERN CALIBRATI SUI BANDI REALI
# ═════════════════════════════════════════════════════════════════════════════

PATTERNS = {
    "cig": [
        r"\bCIG[:\s#.\-]*([A-Z0-9]{10})\b",
        r"codice\s+identificativo\s+(?:gara|CIG)[:\s]*([A-Z0-9]{10})",
    ],
    "cup": [
        r"\bCUP[:\s#.\-]*([A-Z]\d{2}[A-Z]\d{11})\b",
        r"codice\s+unico\s+(?:di\s+)?progetto[:\s]*([A-Z]\d{2}[A-Z]\d{11})",
    ],
    "cpv": [
        r"\b(\d{8}-\d)\b",
        r"(?:CPV|codice\s+CPV)[:\s]*(\d{8}-\d)",
    ],
    "nuts_code": [
        r"\b(IT[A-Z]\d{2,3})\b",
    ],
    "stazione_appaltante": [
        r"(?:Centrale\s+Unica\s+di\s+Committenza)[:\s'\"]+([^\n]{5,120})",
        r"(?:stazione\s+appaltante|committente|ente\s+appaltante|amministrazione\s+aggiudicatrice)[:\s]+([^\n]{5,120})",
        r"(?:Risorse\s+per\s+Roma|Citta\s+Metropolitana|Comune|Provincia|Regione|ASL|Azienda)[^\n]{0,5}(?:di\s+)?([A-Z][^\n]{3,80})",
    ],
    "amministrazione_delegante": [
        r"(?:amministrazione\s+delegante|ente\s+delegante|per\s+conto\s+(?:del|della|di))[:\s]+([^\n]{5,120})",
    ],
    "rup": [
        r"(?:RUP|responsabile\s+unico\s+(?:del\s+)?(?:procedimento|progetto))[:\s]+((?:Ing|Arch|Geom|Dott|Avv)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})(?=[\s\n,\.\-]|$)",
    ],
    "responsabile_procedimento": [
        r"(?:responsabile\s+(?:del\s+)?procedimento)[:\s]+((?:Ing|Arch|Geom|Dott|Avv)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
    ],
    "oggetto_appalto": [
        r"(?:oggetto(?:\s+dell[ao])?\s+(?:gara|appalto|contratto|affidamento|incarico|procedura))[:\s]+([^\n]{15,400})",
        r"(?:affidamento(?:\s+(?:di|dei|dell[ao]))?)[:\s]+([^\n]{15,300})",
        r"(?:servizi\s+di\s+ingegneria[^\n]{5,200})",
        r"(?:redazione\s+della\s+progettazione[^\n]{10,200})",
    ],
    "importo_totale": [
        r"(?:importo\s+totale\s+(?:procedura|complessivo|dell[ao]|a\s+base)?)[:\s]*(?:EUR|euro|€)?\s*([0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})?)",
        r"(?:valore\s+(?:totale\s+)?stimato)[:\s]*(?:EUR|euro|€)?\s*([0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})?)",
    ],
    "importo_base_gara": [
        r"(?:importo\s+(?:a\s+)?base\s+(?:di\s+)?(?:gara|asta|appalto))[:\s]*(?:EUR|euro|€)?\s*([0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})?)",
        r"(?:base\s+d[i']\s*(?:gara|asta))[:\s]*(?:EUR|euro|€)?\s*([0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})?)",
    ],
    "quota_ribassabile": [
        r"(?:35\s*%|quota\s+(?:disponibile|ribassabile))[^\n]{0,40}?([0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})?)",
        r"(?:soggett[oa]\s+a\s+ribasso)[^\n]{0,40}?([0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})?)",
    ],
    "garanzia_provvisoria": [
        r"(?:garanzia|cauzione)\s+provvisoria[:\s]*(?:EUR|euro|€)?\s*([0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})?(?:\s*%)?)",
    ],
    "garanzia_definitiva": [
        r"(?:garanzia\s+definitiva)[^\n]{0,60}?(\d+\s*%)",
    ],
    "punteggio_tecnica": [
        r"(?:offerta\s+tecnica|qualita)[^\n]{0,30}?(\d+)\s*(?:punti?|pt|p\.ti)",
        r"(\d+)\s*punti?\s+(?:all[ao]?\s+)?(?:offerta\s+)?tecnic[ao]",
    ],
    "punteggio_economica": [
        r"(?:offerta\s+economica|prezzo)[^\n]{0,30}?(\d+)\s*(?:punti?|pt|p\.ti)",
        r"(\d+)\s*punti?\s+(?:all[ao]?\s+)?(?:offerta\s+)?economic[ao]",
    ],
    "soglia_sbarramento": [
        r"(?:soglia\s+(?:di\s+)?(?:sbarramento|minima|ammissione)|punteggio\s+minimo\s+tecnico)[:\s]*(\d+)\s*(?:punti?)?",
        r"(?:almeno|minimo)\s+(\d+)\s+punti?\s+(?:tecnici?|qualita)",
    ],
    "giovani_professionisti": [
        r"(?:giovani?\s+professionisti?[^\n]{0,80}?)(\d+)\s*(?:punti?|pt)",
        r"iscritti?\s+all[' ](?:albo|ordine)\s+da\s+meno\s+di\s+5\s+anni[^\n]{0,80}?(\d+)\s*(?:punti?)",
    ],
    "iso_9001": [
        r"ISO\s*9001[^\n]{0,40}?(\d+)\s*(?:punti?|pt)",
        r"(\d+)\s*(?:punti?)\s+(?:per\s+)?ISO\s*9001",
    ],
    "parita_genere": [
        r"(?:parita\s+di\s+genere|certificazione\s+parita)[^\n]{0,60}?(\d+)\s*(?:punti?|pt)",
        r"(?:art\.\s*46.bis)[^\n]{0,60}?(\d+)\s*(?:punti?)",
    ],
    "scadenza_offerte": [
        r"(?:scadenza|offerte?\s+entro)[^\n]{0,60}?(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}[^\n]{0,25})",
        r"(?:offerte?\s+entro)[:\s]*(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}(?:\s+ore?\s+\d{1,2}[:.]\d{2})?)",
        r"FASE\s+1[^\n]{0,40}?(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}(?:\s+ore?\s+\d{1,2}[:.]\d{2})?)",
    ],
    "termine_chiarimenti": [
        r"(?:termine[^\n]{0,30}chiarimenti|chiarimenti[^\n]{0,20}entro)[:\s]*(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}(?:\s+ore?\s+\d{1,2}[:.]\d{2})?)",
    ],
    "durata_contratto": [
        r"(?:durata\s+(?:(?:dell[ao]\s+)?(?:contratto|accordo\s+quadro|servizio|incarico)))[:\s]*(\d+\s*(?:mesi?|anni?|giorni?)(?:[^\n]{0,40})?)",
        r"(\d+)\s+giorni\s+natural[i]?\s+e\s+consecutivi",
    ],
    "piattaforma_url": [
        r"(https?://[^\s\n\"'<>]{15,200})",
    ],
    "fatturato_minimo": [
        r"(?:importo\s+minimo)\s*(?:euro)?\s*([0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})?)",
        r"(?:fatturato\s+(?:globale|specifico|minimo)[^\n]{0,80})([0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})?)\s*(?:euro|EUR)?",
    ],
    "periodo_anni": [
        r"(?:ultimi?\s+)(\d+)\s+(?:anni?|esercizi?)",
    ],
    "contributo_anac": [
        r"(?:contributo\s+ANAC|contributo\s+(?:di\s+)?gara)[:\s]*(?:euro|EUR|€)?\s*([0-9]{1,5}[,.]?\d{0,2})",
    ],
    "imposta_bollo": [
        r"(?:imposta\s+(?:di\s+)?bollo)[:\s]*(?:euro|EUR|€)?\s*([0-9]{1,3}[,.]\d{2})",
    ],
    "subappalto": [
        r"(?:subappalto)[:\s]+([^\n]{10,200})",
    ],
    "avvalimento": [
        r"(?:avvalimento)[:\s]+([^\n]{10,150})",
    ],
    "numero_lotti": [
        r"(?:suddivisa?\s+in|articolat[ao]\s+in)\s+(\d+|due|tre|quattro|cinque)\s+lott",
    ],
    "categorie_ingegneria": [
        r"\b(E\.?\d{2}|S\.?\d{2}|IA\.?\d{2}|IB\.?\d{2}|D\.?\d{2})\b",
    ],
    "verifica_anomalia": [
        r"(offerte?\s+con\s+punti?\s+(?:tecnici?|qualitativi?)[^\n]{0,150})",
        r"(esclusione\s+automatica\s+delle\s+offerte\s+anomal[^\n]{0,60})",
        r"(soglia\s+di\s+anomalia[^\n]{0,80})",
    ],
}

# ═════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═════════════════════════════════════════════════════════════════════════════

def init_db():
    conn = sqlite3.connect(DB_PATH)
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
            score REAL DEFAULT 0,
            feedback TEXT
        );
        CREATE TABLE IF NOT EXISTS training_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            field TEXT NOT NULL,
            text_snippet TEXT NOT NULL,
            correct_value TEXT NOT NULL,
            wrong_value TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            field TEXT,
            version INTEGER,
            accuracy REAL,
            samples_count INTEGER,
            trained_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS feedback_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT,
            field TEXT,
            original TEXT,
            corrected TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS auto_learn_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT,
            field TEXT,
            value TEXT,
            confidence REAL,
            validation_ok INTEGER,
            action TEXT,
            reason TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS quarantine (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT,
            field TEXT,
            value TEXT,
            snippet TEXT,
            confidence REAL,
            status TEXT DEFAULT 'pending',
            reviewed_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    # Migrations
    for col, tbl in [('full_text', 'documents')]:
        try: c.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} TEXT")
        except: pass
    conn.commit()
    conn.close()

# ═════════════════════════════════════════════════════════════════════════════
# ESTRATTORE
# ═════════════════════════════════════════════════════════════════════════════

class AppaltoExtractor:

    def __init__(self):
        init_db()
        self.ml_models = {}
        self.load_ml_models()

    def clean(self, s):
        if not s:
            return s
        s = re.sub(r'\s+', ' ', s).strip()
        s = re.sub(r'[,;:\s]+$', '', s)
        return s

    def normalize_amount(self, raw):
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
        except:
            return raw

    def first_match(self, text, patterns):
        for pat in patterns:
            try:
                m = re.search(pat, text, re.I | re.M)
                if m:
                    return self.clean(m.group(1))
            except:
                continue
        return None

    def all_matches(self, text, patterns):
        found = []
        for pat in patterns:
            try:
                found.extend(re.findall(pat, text, re.I | re.M))
            except:
                continue
        return list(dict.fromkeys([self.clean(x) for x in found if x]))

    def extract_int(self, text, patterns):
        val = self.first_match(text, patterns)
        if val:
            try:
                return int(re.sub(r'\D', '', val))
            except:
                return None
        return None

    def parse_number_word(self, s):
        if not s:
            return None
        mapping = {"due": 2, "tre": 3, "quattro": 4, "cinque": 5, "sei": 6}
        sl = s.strip().lower()
        if sl in mapping:
            return mapping[sl]
        try:
            return int(re.sub(r'\D', '', s))
        except:
            return None

    # ── Context / Snippet finder ───────────────────────────────────────────

    def _find_value_context(self, text, value, window=300):
        """Trova il contesto testuale dove appare un valore nel documento.
        Ritorna il frammento di testo circostante per il training ML."""
        if not value or not text:
            return ""
        val_str = str(value).strip()
        if len(val_str) < 2:
            return ""
        # Ricerca diretta
        idx = text.lower().find(val_str.lower())
        if idx >= 0:
            start = max(0, idx - window)
            end = min(len(text), idx + len(val_str) + window)
            return text[start:end].strip()
        # Ricerca parole significative
        words = [w for w in val_str.split() if len(w) > 3]
        for word in words:
            idx = text.lower().find(word.lower())
            if idx >= 0:
                start = max(0, idx - window)
                end = min(len(text), idx + len(word) + window)
                return text[start:end].strip()
        # Ricerca numeri (per importi)
        nums = re.findall(r'\d{3,}', val_str)
        for num in nums:
            idx = text.find(num)
            if idx >= 0:
                start = max(0, idx - window)
                end = min(len(text), idx + len(num) + window)
                return text[start:end].strip()
        return ""

    def get_document_text(self, doc_id):
        """Recupera testo completo e dati di un documento salvato"""
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        row = c.execute(
            "SELECT full_text, extracted_json, corrected_json FROM documents WHERE id=?",
            (doc_id,)
        ).fetchone()
        conn.close()
        if row:
            return {
                "text": row[0] or "",
                "extracted": json.loads(row[1]) if row[1] else {},
                "corrected": json.loads(row[2]) if row[2] else {}
            }
        return None

    # ── Classificatori ────────────────────────────────────────────────────

    def classify_procedure(self, text):
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
        return "Non specificata"

    def classify_criterio(self, text):
        if re.search(r'offerta\s+economicamente\s+piu\s+vantaggiosa|OEPV', text, re.I):
            art = re.search(r'art(?:icolo)?\.?\s*108\s+comma\s+\d+', text, re.I)
            suffix = f" ({self.clean(art.group(0))} D.Lgs. 36/2023)" if art else " (art. 108 D.Lgs. 36/2023)"
            return "OEPV — Offerta Economicamente Più Vantaggiosa" + suffix
        if re.search(r'(?:massimo|minor|minimo)\s+ribasso', text, re.I):
            return "Massimo/Minor Ribasso"
        if re.search(r'prezzo\s+piu\s+basso', text, re.I):
            return "Prezzo più basso"
        return "Non specificato"

    # ── Estrazioni complesse ───────────────────────────────────────────────

    def extract_lotti_detail(self, text):
        lotti = []
        parts = re.split(r'(?i)\bLotto\s+(\d+)\b', text)
        if len(parts) > 2:
            for i in range(1, len(parts) - 1, 2):
                n = parts[i]
                body = parts[i + 1][:2000]
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

    def extract_categorie_ingegneria(self, text):
        cats = set()
        for m in re.finditer(r'\b(E\.?\d{2}|S\.?\d{2}|IA\.?\d{2}|IB\.?\d{2}|D\.?\d{2})\b', text, re.I):
            cat = m.group(1).upper()
            # Normalizza E21 -> E.21, IA01 -> IA.01
            cat = re.sub(r'^(IA|IB|[ESDG])(\d+)$', lambda x: x.group(1) + '.' + x.group(2), cat)
            cats.add(cat)
        return sorted(cats)

    def extract_criteri_tecnici(self, text):
        criteri = []
        for m in re.finditer(
            r'(?:criterio\s+)?([A-Z](?:\.\d+)?)\s*[-.:]\s*([^\n]{5,120}?)\s+(\d+)\s*(?:punti?|pt|p\.ti)',
            text, re.I | re.M
        ):
            criteri.append({
                "codice": m.group(1),
                "nome": self.clean(m.group(2))[:100],
                "punteggio": int(m.group(3))
            })
        return criteri[:15]

    def extract_struttura_compenso(self, text):
        result = {
            "quota_fissa_65_perc": bool(re.search(r'65\s*%[^\n]{0,60}(?:fisso|non\s+soggett[oa]|prezzo\s+fisso)', text, re.I)),
            "quota_ribassabile_35_perc": bool(re.search(r'35\s*%[^\n]{0,60}(?:ribassabile|soggett[oa]|disponibile)', text, re.I)),
        }
        m35 = re.search(r'35\s*%[^\n]{0,60}?€?\s*([0-9]{1,3}(?:[.,][0-9]{3})*[.,]\d{2})', text, re.I)
        if m35:
            result["importo_ribassabile"] = self.normalize_amount(m35.group(1))
        return result

    def extract_scadenze(self, text):
        scadenze = {}
        main = self.first_match(text, PATTERNS["scadenza_offerte"])
        if main:
            scadenze["principale"] = main
        m_fase = re.search(r'FASE\s+1[^\n]{0,40}?(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}[^\n]{0,30})', text, re.I)
        if m_fase:
            scadenze["fase_1_archivi"] = self.clean(m_fase.group(1))
        m_pub = re.search(r'(?:prima\s+)?sessione\s+pubblica[^\n]{0,30}?(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}[^\n]{0,20})', text, re.I)
        if m_pub:
            scadenze["sessione_pubblica"] = self.clean(m_pub.group(1))
        return scadenze

    def extract_vincoli_lotti(self, text):
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

    def extract_sopralluogo(self, text):
        obbligatorio = bool(re.search(r'sopralluogo\s+(?:e\s+)?obbligatorio|sopralluogo\s+delle?\s+aree', text, re.I))
        non_previsto = bool(re.search(r'sopralluogo[^\n]{0,30}non\s+(?:prev|richiesto)', text, re.I))
        m = re.search(r'sopralluogo[^\n]{0,300}', text, re.I)
        return {
            "obbligatorio": obbligatorio and not non_previsto,
            "note": self.clean(m.group(0))[:250] if m else None
        }

    def extract_note_operative(self, text):
        note = []
        checks = [
            (r'sopralluogo\s+(?:e\s+)?obbligatorio|sopralluogo\s+delle?\s+aree',
             "🚶 Sopralluogo obbligatorio — programmare la visita prima della scadenza"),
            (r'inversione\s+procedimentale',
             "🔄 Inversione procedimentale (art. 107 c.3) — documentazione valutata prima dell'offerta economica"),
            (r'soglia\s+(?:di\s+)?sbarramento',
             "⛔ Soglia di sbarramento tecnica — verificare punteggio minimo per accedere alla fase economica"),
            (r'giovani?\s+professionisti?',
             "👨‍🎓 Bonus giovani professionisti (iscritti albo <5 anni) — può fare differenza nel punteggio tabellare"),
            (r'parita\s+di\s+genere|certificazione\s+parita',
             "♀♂ Certificazione parità di genere (art. 46-bis) — punteggio bonus disponibile"),
            (r'ISO\s*9001',
             "📋 Bonus certificazione ISO 9001 — verificare se già in possesso"),
            (r'vincolo.{0,30}(?:entrambi|tutti)\s+(?:i\s+)?lotti',
             "🔗 Partecipazione obbligatoria a tutti i lotti"),
            (r'offerta\s+identica.{0,30}(?:entrambi|tutti)',
             "📋 Offerta identica per tutti i lotti — ribasso unico"),
            (r'accordo\s+quadro.{0,30}unico\s+operatore',
             "📑 Accordo Quadro a unico operatore — possibili chiamate successive senza nuova gara"),
            (r'conformita.{0,10}CAM|criteri\s+ambientali\s+minimi',
             "🌿 Conformità CAM obbligatoria — verificare requisiti D.M. specifico"),
            (r'DUVRI.{0,60}non\s+(?:prev|ricorr)|rischi\s+interferenze.{0,40}non\s+ricorr',
             "✅ DUVRI non previsto — natura intellettuale del servizio"),
            (r'costi\s+(?:della\s+)?manodopera.{0,30}pari\s+a\s+.?\s*0',
             "ℹ️ Costi manodopera = €0 (conforme a servizi intellettuali)"),
            (r'(?:FESR|PNRR|PON|POR|PR\s+Campania|decreto\s+dirigenziale)',
             "🇪🇺 Finanziamento pubblico/europeo — potrebbero esserci clausole specifiche di rendicontazione"),
            (r'soccorso\s+istruttorio',
             "📋 Soccorso istruttorio previsto — documenti mancanti sanabili entro termine"),
            (r'self[\s-]?cleaning',
             "🔄 Self-cleaning ammesso — possibile partecipazione nonostante cause ostative"),
            (r'quota.{0,10}PMI.{0,30}(?:20|30)\s*%',
             "🏢 Quota subappalto PMI obbligatoria — prevedere coinvolgimento piccole imprese"),
            (r'inversione\s+fasi|prima\s+le\s+buste',
             "⚠️ Verifica ordine apertura buste — procedura con eventuale inversione fasi"),
        ]
        for pattern, nota in checks:
            if re.search(pattern, text, re.I):
                note.append(nota)
        return note

    def detect_finanziamento(self, text):
        fonti = {
            "PNRR": r'PNRR|Piano\s+Nazionale\s+di\s+Ripresa',
            "FESR": r'FESR|Fondo\s+Europeo\s+di\s+Sviluppo\s+Regionale',
            "PON/POR": r'(?:PON|POR|PR)\s+[A-Za-z\s]+20\d{2}',
            "Decreto Dirigenziale/Ministeriale": r'decreto\s+(?:dirigenziale|ministeriale)[^\n]{0,60}',
            "Fondi propri": r'(?:bilancio|fondi)\s+(?:propri|comunali|regionali|metropolitani)',
        }
        for nome, pat in fonti.items():
            if re.search(pat, text, re.I):
                return nome
        return None

    # ── ESTRAZIONE PRINCIPALE ─────────────────────────────────────────────

    def extract(self, text: str, filename: str = "") -> dict:
        r = {}

        # Identificativi
        r["cig"] = self.first_match(text, PATTERNS["cig"])
        cups = self.all_matches(text, PATTERNS["cup"])
        r["cup"] = cups[0] if len(cups) == 1 else (cups if cups else None)
        cpvs = self.all_matches(text, PATTERNS["cpv"])
        r["cpv"] = cpvs[0] if len(cpvs) == 1 else (cpvs if cpvs else None)
        r["nuts_code"] = self.first_match(text, PATTERNS["nuts_code"])

        # Soggetti
        r["stazione_appaltante"] = self.first_match(text, PATTERNS["stazione_appaltante"])
        r["amministrazione_delegante"] = self.first_match(text, PATTERNS["amministrazione_delegante"])
        r["rup"] = self.first_match(text, PATTERNS["rup"])
        r["responsabile_procedimento"] = self.first_match(text, PATTERNS["responsabile_procedimento"])

        # Oggetto e procedura
        r["oggetto_appalto"] = self.first_match(text, PATTERNS["oggetto_appalto"])
        r["tipo_procedura"] = self.classify_procedure(text)
        r["criterio_aggiudicazione"] = self.classify_criterio(text)

        # Accordo quadro
        r["is_accordo_quadro"] = bool(re.search(r'accordo\s+quadro', text, re.I))
        if r["is_accordo_quadro"]:
            m = re.search(r'accordo\s+quadro\s+(?:a|con)\s+(unico|doppio)', text, re.I)
            r["tipo_accordo_quadro"] = self.clean(m.group(1)) if m else None

        # Lotti
        n_raw = self.first_match(text, PATTERNS["numero_lotti"])
        r["numero_lotti"] = self.parse_number_word(n_raw)
        r["lotti"] = self.extract_lotti_detail(text)
        r["vincoli_lotti"] = self.extract_vincoli_lotti(text)

        # Importi
        r["importo_totale"] = self.normalize_amount(self.first_match(text, PATTERNS["importo_totale"]))
        r["importo_base_gara"] = self.normalize_amount(self.first_match(text, PATTERNS["importo_base_gara"]))
        r["struttura_compenso_65_35"] = self.extract_struttura_compenso(text)
        r["garanzia_provvisoria"] = self.normalize_amount(self.first_match(text, PATTERNS["garanzia_provvisoria"]))
        r["garanzia_definitiva"] = self.first_match(text, PATTERNS["garanzia_definitiva"])

        # Punteggi OEPV
        r["punteggio_tecnica"] = self.extract_int(text, PATTERNS["punteggio_tecnica"])
        r["punteggio_economica"] = self.extract_int(text, PATTERNS["punteggio_economica"])
        r["soglia_sbarramento_tecnica"] = self.extract_int(text, PATTERNS["soglia_sbarramento"])
        r["criteri_tecnici"] = self.extract_criteri_tecnici(text)

        # Bonus tabellari
        r["pti_giovani_professionisti"] = self.extract_int(text, PATTERNS["giovani_professionisti"])
        r["pti_iso_9001"] = self.extract_int(text, PATTERNS["iso_9001"])
        r["pti_parita_genere"] = self.extract_int(text, PATTERNS["parita_genere"])

        # Tempistiche
        r["scadenze"] = self.extract_scadenze(text)
        r["termine_chiarimenti"] = self.first_match(text, PATTERNS["termine_chiarimenti"])
        r["durata_contratto"] = self.first_match(text, PATTERNS["durata_contratto"])

        # Categorie tecniche ingegneria
        r["categorie_ingegneria"] = self.extract_categorie_ingegneria(text)

        # Requisiti
        r["periodo_requisiti_anni"] = self.extract_int(text, PATTERNS["periodo_anni"])
        r["fatturato_minimo"] = self.normalize_amount(self.first_match(text, PATTERNS["fatturato_minimo"]))

        # Sopralluogo
        r["sopralluogo"] = self.extract_sopralluogo(text)

        # Piattaforma
        r["piattaforma_url"] = self.first_match(text, PATTERNS["piattaforma_url"])

        # Contributi
        r["contributo_anac"] = self.normalize_amount(self.first_match(text, PATTERNS["contributo_anac"]))
        r["imposta_bollo"] = self.normalize_amount(self.first_match(text, PATTERNS["imposta_bollo"]))

        # Subappalto/Avvalimento
        r["subappalto"] = self.first_match(text, PATTERNS["subappalto"])
        r["avvalimento"] = self.first_match(text, PATTERNS["avvalimento"])

        # Flags booleani
        r["revisione_prezzi"] = bool(re.search(r'revisione\s+(?:dei\s+)?prezzi', text, re.I))
        r["conformita_cam"] = bool(re.search(r'\bCAM\b|criteri\s+ambientali\s+minimi', text, re.I))
        r["inversione_procedimentale"] = bool(re.search(r'inversione\s+procedimentale', text, re.I))

        # Anomalia
        r["verifica_anomalia"] = self.first_match(text, PATTERNS["verifica_anomalia"])

        # Finanziamento
        r["finanziamento"] = self.detect_finanziamento(text)

        # Note operative
        r["note_operative"] = self.extract_note_operative(text)

        # ── Traccia snippet e metodo per ogni campo ──────────────────────
        _snippets = {}
        _methods = {}
        for key, value in r.items():
            if key.startswith('_') or value is None:
                continue
            if isinstance(value, bool):
                if value:
                    _methods[key] = "rules"
                continue
            if isinstance(value, dict):
                # Oggetti complessi (scadenze, vincoli, etc)
                has_content = any(v for v in value.values() if v not in [None, False, ""])
                if has_content:
                    _methods[key] = "rules"
                continue
            if isinstance(value, list):
                if len(value) > 0:
                    _methods[key] = "rules"
                    first = str(value[0]) if value else ""
                    ctx = self._find_value_context(text, first)
                    if ctx:
                        _snippets[key] = ctx
                continue
            if value in ["", 0]:
                continue
            _methods[key] = "rules"
            ctx = self._find_value_context(text, str(value))
            if ctx:
                _snippets[key] = ctx

        # ── ML fallback — esteso a TUTTI i campi testuali ─────────────────
        ml_candidates = [
            k for k, v in r.items()
            if not k.startswith('_')
            and v in [None, "", 0]
            and not isinstance(v, (bool, dict, list))
            and k in self.ml_models
        ]
        for field in ml_candidates:
            pred = self._ml_predict(field, text[:3000])
            if pred:
                r[field] = pred
                _methods[field] = "ml"
                ctx = self._find_value_context(text, str(pred))
                if ctx:
                    _snippets[field] = ctx

        r["_snippets"] = _snippets
        r["_methods"] = _methods

        # ── Confidence ────────────────────────────────────────────────────
        empty_objects = [
            {"quota_fissa_65_perc": False, "quota_ribassabile_35_perc": False},
            {"obbligatorio": False, "note": None},
            {"partecipazione_tutti_lotti": False, "offerta_identica": False,
             "max_lotti_aggiudicazione": None, "medesima_forma_giuridica": False}
        ]
        total = sum(1 for k in r if not k.startswith('_'))
        filled = sum(
            1 for k, v in r.items()
            if not k.startswith('_')
            and v not in [None, [], {}, False, "", 0]
            and v not in empty_objects
        )
        r["_confidence"] = round(filled / max(total, 1) * 100, 1)
        r["_extraction_method"] = "hybrid" if any(m == "ml" for m in _methods.values()) else "rules"
        r["_timestamp"] = datetime.now().isoformat()

        # Salva e auto-learn
        doc_id = hashlib.sha256((filename + text[:500]).encode()).hexdigest()[:16]
        self._save_document(doc_id, filename, text, r)
        self._auto_learn(r, text)
        r["_doc_id"] = doc_id

        return r

    # ═════════════════════════════════════════════════════════════════════
    # SAFE AUTO-LEARNING — apprendimento automatico con protezione errori
    # ═════════════════════════════════════════════════════════════════════

    def _validate_field(self, field, value):
        """Valida un valore estratto con il validatore specifico del campo.
        Ritorna True se il valore è nel formato corretto."""
        validator = FIELD_VALIDATORS.get(field)
        if not validator:
            return True  # Nessun validatore = accetta (per campi booleani/flag)
        try:
            return validator(value)
        except:
            return False

    def _per_field_confidence(self, field, value, method, text):
        """Calcola la confidenza per singolo campo (0.0 - 1.0).
        Combina: metodo di estrazione + validazione formato + qualità snippet."""
        score = 0.0

        # 1. Metodo di estrazione
        if method == "rules":
            score += 0.55  # Regex match = buona base
        elif method == "ml":
            score += 0.35  # ML prediction = base media
        elif method == "corrected":
            score += 0.95  # Correzione umana = massima fiducia
            return min(score, 1.0)

        # 2. Validazione formato
        if self._validate_field(field, value):
            score += 0.25  # Formato valido = +0.25
        else:
            score -= 0.20  # Formato invalido = penalità

        # 3. Campi con formato rigido (CIG, CUP, CPV) → bonus se regex li ha trovati
        rigid_fields = {'cig', 'cup', 'cpv', 'nuts_code'}
        if field in rigid_fields and method == "rules":
            score += 0.15  # Formato rigido + regex = alta affidabilità

        # 4. Punteggi sensibili: se tecnica + economica sommano ~100 → bonus
        if field in ('punteggio_tecnica', 'punteggio_economica'):
            score += 0.05  # I punteggi regex sono tipicamente affidabili

        # 5. Snippet lungo → l'AI ha trovato il contesto → più affidabile
        snippet = self._find_value_context(text, str(value), window=200)
        if snippet and len(snippet) > 50:
            score += 0.10

        return max(0.0, min(score, 1.0))

    def _auto_learn(self, result, text):
        """Safe auto-learning: impara automaticamente solo da estrazioni
        che superano validazione E confidenza alta.
        Risultati dubbi vanno in quarantena per review umano."""
        methods = result.get("_methods", {})
        snippets = result.get("_snippets", {})
        doc_id = result.get("_doc_id", "")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        auto_learned = 0
        quarantined = 0

        for field, value in result.items():
            # Skip metadata e campi complessi
            if field.startswith('_') or field in AUTO_LEARN_BLACKLIST:
                continue
            if value is None or value == "" or value == 0:
                continue
            if isinstance(value, (bool, dict, list)):
                continue

            method = methods.get(field, "")
            val_str = str(value)

            # ── Step 1: Validazione formato ─────────────────────────
            is_valid = self._validate_field(field, value)

            # ── Step 2: Calcolo confidenza per campo ────────────────
            confidence = self._per_field_confidence(field, value, method, text)

            # ── Step 3: Anti-duplicazione ───────────────────────────
            existing = c.execute(
                "SELECT COUNT(*) FROM training_samples WHERE field=? AND correct_value=?",
                (field, val_str)
            ).fetchone()[0]
            if existing > 0:
                # Già presente → skip (non duplicare)
                c.execute(
                    "INSERT INTO auto_learn_log (doc_id, field, value, confidence, validation_ok, action, reason) VALUES (?,?,?,?,?,?,?)",
                    (doc_id, field, val_str, confidence, int(is_valid), "skip", "duplicato_esistente")
                )
                continue

            # ── Step 4: Trova snippet contesto ──────────────────────
            snippet = snippets.get(field, "")
            if not snippet:
                snippet = self._find_value_context(text, val_str, window=400)
            if not snippet or len(snippet) < 20:
                snippet = text[:1500]  # Fallback minimo

            # ── Step 5: Decisione AUTO-LEARN / QUARANTENA / SKIP ────
            if is_valid and confidence >= AUTO_LEARN_THRESHOLD:
                # ✅ ALTO → auto-learn direttamente
                c.execute(
                    "INSERT INTO training_samples (field, text_snippet, correct_value) VALUES (?,?,?)",
                    (field, snippet[:2000], val_str)
                )
                c.execute(
                    "INSERT INTO auto_learn_log (doc_id, field, value, confidence, validation_ok, action, reason) VALUES (?,?,?,?,?,?,?)",
                    (doc_id, field, val_str, confidence, 1, "auto_learned",
                     f"conf={confidence:.2f}, valid=True, method={method}")
                )
                auto_learned += 1

            elif is_valid and confidence >= QUARANTINE_THRESHOLD:
                # ⚠️ MEDIO → quarantena (review umano)
                c.execute(
                    "INSERT INTO quarantine (doc_id, field, value, snippet, confidence) VALUES (?,?,?,?,?)",
                    (doc_id, field, val_str, snippet[:2000], confidence)
                )
                c.execute(
                    "INSERT INTO auto_learn_log (doc_id, field, value, confidence, validation_ok, action, reason) VALUES (?,?,?,?,?,?,?)",
                    (doc_id, field, val_str, confidence, 1, "quarantined",
                     f"conf={confidence:.2f} < {AUTO_LEARN_THRESHOLD} threshold")
                )
                quarantined += 1

            else:
                # ❌ BASSO / non valido → skip
                c.execute(
                    "INSERT INTO auto_learn_log (doc_id, field, value, confidence, validation_ok, action, reason) VALUES (?,?,?,?,?,?,?)",
                    (doc_id, field, val_str, confidence, int(is_valid), "skipped",
                     f"conf={confidence:.2f}, valid={is_valid}")
                )

        conn.commit()

        # ── Step 6: Trigger training se abbastanza campioni ─────────
        fields_to_check = set(f for f, _ in methods.items() if f not in AUTO_LEARN_BLACKLIST)
        for field in fields_to_check:
            count = c.execute("SELECT COUNT(*) FROM training_samples WHERE field=?", (field,)).fetchone()[0]
            if count >= 5 and count % 3 == 0:
                self._safe_train_with_rollback(field)

        conn.close()
        return {"auto_learned": auto_learned, "quarantined": quarantined}

    # ── Quarantine management ──────────────────────────────────────────
    def get_quarantine(self, limit=50):
        """Restituisce i campioni in quarantena per review umano."""
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        rows = c.execute(
            "SELECT id, doc_id, field, value, snippet, confidence, status, created_at "
            "FROM quarantine WHERE status='pending' ORDER BY confidence DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        return [
            {"id": r[0], "doc_id": r[1], "field": r[2], "value": r[3],
             "snippet": r[4], "confidence": r[5], "status": r[6], "created_at": r[7]}
            for r in rows
        ]

    def approve_quarantine(self, qid):
        """Approva un campione dalla quarantena → diventa training sample."""
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        row = c.execute("SELECT field, value, snippet FROM quarantine WHERE id=? AND status='pending'", (qid,)).fetchone()
        if not row:
            conn.close()
            return {"status": "error", "message": "Campione non trovato o gia' processato"}
        field, value, snippet = row
        # Aggiungi al training
        c.execute("INSERT INTO training_samples (field, text_snippet, correct_value) VALUES (?,?,?)",
                  (field, snippet, value))
        c.execute("UPDATE quarantine SET status='approved', reviewed_at=? WHERE id=?",
                  (datetime.now().isoformat(), qid))
        conn.commit()
        # Check training trigger
        count = c.execute("SELECT COUNT(*) FROM training_samples WHERE field=?", (field,)).fetchone()[0]
        conn.close()
        retrained = False
        if count >= 5 and count % 3 == 0:
            self._safe_train_with_rollback(field)
            retrained = True
        return {"status": "ok", "field": field, "sample_count": count, "retrained": retrained}

    def reject_quarantine(self, qid):
        """Rifiuta un campione dalla quarantena → non entra nel training."""
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE quarantine SET status='rejected', reviewed_at=? WHERE id=?",
                  (datetime.now().isoformat(), qid))
        conn.commit()
        conn.close()
        return {"status": "ok"}

    def get_auto_learn_stats(self):
        """Statistiche sull'auto-learning: quanti auto-learned, quarantined, skipped."""
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        stats = {}
        for action in ('auto_learned', 'quarantined', 'skipped', 'skip'):
            n = c.execute("SELECT COUNT(*) FROM auto_learn_log WHERE action=?", (action,)).fetchone()[0]
            stats[action] = n
        pending = c.execute("SELECT COUNT(*) FROM quarantine WHERE status='pending'").fetchone()[0]
        stats["quarantine_pending"] = pending
        # Per-field breakdown
        field_stats = c.execute(
            "SELECT field, action, COUNT(*) FROM auto_learn_log GROUP BY field, action"
        ).fetchall()
        by_field = {}
        for f, a, n in field_stats:
            if f not in by_field:
                by_field[f] = {}
            by_field[f][a] = n
        stats["by_field"] = by_field
        conn.close()
        return stats

    # ── ML ────────────────────────────────────────────────────────────────
    def load_ml_models(self):
        for mf in MODEL_DIR.glob("model_*.pkl"):
            field = mf.stem.replace("model_", "")
            try:
                with open(mf, "rb") as f:
                    self.ml_models[field] = pickle.load(f)
            except:
                pass

    def _ml_predict(self, field, snippet):
        model = self.ml_models.get(field)
        if model and snippet:
            try:
                return model.predict([snippet])[0]
            except:
                return None
        return None

    def train_field_classifier(self, field):
        return self._safe_train_with_rollback(field)

    def _safe_train_with_rollback(self, field):
        """Training con protezione: testa il nuovo modello su holdout.
        Se performa peggio del precedente, fa rollback."""
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        rows = c.execute(
            "SELECT text_snippet, correct_value FROM training_samples WHERE field=?", (field,)
        ).fetchall()
        conn.close()

        if len(rows) < 3:
            return None, f"Campioni insufficienti ({len(rows)}) per '{field}'. Servono almeno 3."

        texts = [r[0] for r in rows]
        labels = [r[1] for r in rows]

        # ── Holdout validation (se abbastanza campioni) ──────────────
        old_model = self.ml_models.get(field)
        old_accuracy = None
        new_accuracy = None

        if len(rows) >= 6:
            # Split 80/20 per validazione
            import random
            indices = list(range(len(rows)))
            random.seed(42)
            random.shuffle(indices)
            split = max(2, int(len(indices) * 0.2))
            test_idx = set(indices[:split])
            train_idx = [i for i in indices if i not in test_idx]

            train_texts = [texts[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            test_texts = [texts[i] for i in test_idx]
            test_labels = [labels[i] for i in test_idx]

            # Solo se abbiamo almeno 2 classi nel train set
            if len(set(train_labels)) >= 2 or len(set(labels)) == 1:
                # Testa vecchio modello
                if old_model:
                    try:
                        old_preds = old_model.predict(test_texts)
                        old_accuracy = sum(1 for p, t in zip(old_preds, test_labels) if p == t) / len(test_labels)
                    except:
                        old_accuracy = None

                # Allena nuovo su train set
                try:
                    test_pipeline = Pipeline([
                        ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=8000, analyzer="char_wb")),
                        ("clf", SGDClassifier(loss="hinge", max_iter=1000, random_state=42)),
                    ])
                    test_pipeline.fit(train_texts, train_labels)
                    new_preds = test_pipeline.predict(test_texts)
                    new_accuracy = sum(1 for p, t in zip(new_preds, test_labels) if p == t) / len(test_labels)
                except:
                    new_accuracy = None

                # ── ROLLBACK CHECK ──────────────────────────────────
                if old_accuracy is not None and new_accuracy is not None:
                    if new_accuracy < old_accuracy - 0.05:  # Tolleranza 5%
                        # Nuovo modello È PEGGIO → NON lo salviamo
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        c.execute(
                            "INSERT INTO auto_learn_log (doc_id, field, value, confidence, validation_ok, action, reason) VALUES (?,?,?,?,?,?,?)",
                            ("system", field, "", new_accuracy, 0, "rollback",
                             f"new_acc={new_accuracy:.2f} < old_acc={old_accuracy:.2f}, modello NON aggiornato")
                        )
                        conn.commit()
                        conn.close()
                        return old_model, f"⚠ Rollback: nuovo modello '{field}' ({new_accuracy:.0%}) peggio del precedente ({old_accuracy:.0%}). Mantenuto vecchio."

        # ── Training finale su TUTTI i dati ──────────────────────────
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=8000, analyzer="char_wb")),
            ("clf", SGDClassifier(loss="hinge", max_iter=1000, random_state=42)),
        ])
        pipeline.fit(texts, labels)

        # Backup vecchio modello prima di sovrascrivere
        path = MODEL_DIR / f"model_{field}.pkl"
        backup_path = MODEL_DIR / f"model_{field}_prev.pkl"
        if path.exists():
            import shutil
            shutil.copy2(path, backup_path)

        with open(path, "wb") as f:
            pickle.dump(pipeline, f)
        self.ml_models[field] = pipeline

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        acc_str = f"{new_accuracy:.2f}" if new_accuracy else "N/A"
        c.execute("INSERT INTO model_versions (field, version, accuracy, samples_count, trained_at) VALUES (?,1,?,?,?)",
                  (field, new_accuracy, len(rows), datetime.now().isoformat()))
        conn.commit()
        conn.close()

        msg = f"Modello '{field}' addestrato su {len(rows)} campioni"
        if new_accuracy:
            msg += f" (accuracy holdout: {new_accuracy:.0%})"
        if old_accuracy and new_accuracy:
            msg += f" [prima: {old_accuracy:.0%}]"
        return pipeline, msg

    def record_correction(self, doc_id, field, original, corrected, snippet=""):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # ── Recupera testo completo per trovare contesto intelligente ────
        row_text = c.execute("SELECT full_text FROM documents WHERE id=?", (doc_id,)).fetchone()
        full_text = row_text[0] if row_text and row_text[0] else ""

        # Cerca il contesto migliore per il training
        training_snippet = snippet
        if full_text:
            # Prima: cerca il valore CORRETTO nel testo del PDF
            ctx = self._find_value_context(full_text, corrected, window=500)
            if ctx and len(ctx) > 20:
                training_snippet = ctx
            elif original:
                # Fallback: cerca il valore ORIGINALE (errato)
                ctx = self._find_value_context(full_text, original, window=500)
                if ctx and len(ctx) > 20:
                    training_snippet = ctx

        if not training_snippet:
            training_snippet = full_text[:1500] if full_text else ""

        c.execute("INSERT INTO feedback_log (doc_id, field, original, corrected) VALUES (?,?,?,?)",
                  (doc_id, field, original, corrected))
        c.execute("INSERT INTO training_samples (field, text_snippet, correct_value, wrong_value) VALUES (?,?,?,?)",
                  (field, training_snippet[:2000], corrected, original))

        # Aggiorna JSON corretto del documento
        row = c.execute("SELECT corrected_json FROM documents WHERE id=?", (doc_id,)).fetchone()
        cd = json.loads(row[0]) if row and row[0] else {}
        cd[field] = corrected
        c.execute("UPDATE documents SET corrected_json=? WHERE id=?",
                  (json.dumps(cd, ensure_ascii=False), doc_id))
        conn.commit()
        conn.close()

        # Soglia più bassa = apprendimento più veloce
        count = self._get_sample_count(field)
        retrained = False
        if count >= 5 and count % 3 == 0:
            self._safe_train_with_rollback(field)
            retrained = True
        return retrained

    def _save_document(self, doc_id, filename, text, extracted):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        text_hash = hashlib.md5(text.encode()).hexdigest()
        c.execute("""
            INSERT OR REPLACE INTO documents (id, filename, text_hash, full_text, upload_date, extracted_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (doc_id, filename, text_hash, text,
              datetime.now().isoformat(),
              json.dumps(extracted, ensure_ascii=False, default=str)))
        conn.commit()
        conn.close()

    def _get_sample_count(self, field):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        n = c.execute("SELECT COUNT(*) FROM training_samples WHERE field=?", (field,)).fetchone()[0]
        conn.close()
        return n

    def get_stats(self):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        docs = c.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        corrections = c.execute("SELECT COUNT(*) FROM feedback_log").fetchone()[0]
        samples = c.execute("SELECT field, COUNT(*) FROM training_samples GROUP BY field").fetchall()
        models = c.execute("SELECT field, accuracy, samples_count, trained_at FROM model_versions ORDER BY trained_at DESC").fetchall()
        # Auto-learn stats
        auto_learned = c.execute("SELECT COUNT(*) FROM auto_learn_log WHERE action='auto_learned'").fetchone()[0]
        quarantine_pending = c.execute("SELECT COUNT(*) FROM quarantine WHERE status='pending'").fetchone()[0]
        rollbacks = c.execute("SELECT COUNT(*) FROM auto_learn_log WHERE action='rollback'").fetchone()[0]
        conn.close()
        return {
            "total_documents": docs,
            "total_corrections": corrections,
            "training_samples": {s[0]: s[1] for s in samples},
            "trained_models": [{"field": m[0], "accuracy": m[1], "samples": m[2], "trained_at": m[3]} for m in models],
            "auto_learned": auto_learned,
            "quarantine_pending": quarantine_pending,
            "rollbacks": rollbacks,
        }

    def get_history(self):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        rows = c.execute("""
            SELECT id, filename, upload_date, extracted_json
            FROM documents ORDER BY upload_date DESC LIMIT 50
        """).fetchall()
        conn.close()
        result = []
        for r in rows:
            try:
                data = json.loads(r[3]) if r[3] else {}
                result.append({
                    "id": r[0], "filename": r[1], "date": r[2],
                    "oggetto": data.get("oggetto_appalto", "N/D"),
                    "importo": data.get("importo_totale") or data.get("importo_base_gara", "N/D"),
                    "stazione": data.get("stazione_appaltante", "N/D"),
                    "confidence": data.get("_confidence", 0),
                    "lotti": data.get("numero_lotti"),
                    "procedura": data.get("tipo_procedura", "N/D"),
                })
            except:
                pass
        return result


extractor = AppaltoExtractor()
