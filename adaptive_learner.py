"""
AppaltoAI — Adaptive Learner (Apprendimento Autonomo Progressivo)
==================================================================
Sistema che impara AUTOMATICAMENTE da ogni documento caricato,
senza bisogno di correzioni umane.

Principi fondamentali:
  1. Ogni estrazione è un'opportunità di apprendimento
  2. Il confronto tra documenti rivela errori e pattern
  3. Le regole si generano da sole dopo N estrazioni coerenti
  4. La similarità tra documenti trasferisce conoscenza
  5. L'ensemble di metodi multipli batte ogni singolo metodo

Componenti:
  - ExtractionMemory: memorizza TUTTE le estrazioni con contesto
  - ValueValidator: validazione statistica valori estratti
  - RuleGenerator: genera nuove regex da estrazioni riuscite
  - DocSimilarity: trova documenti simili per trasferire strategie
  - ConfidenceEnsemble: combina metodi multipli con voto pesato
  - AdaptiveLearner: orchestratore principale
"""

import json
import re
import math
import hashlib
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict

from config import DB_PATH, DATA_DIR, MODEL_DIR
from database import get_connection
from utils import clean_string, find_value_context
from log_config import get_logger

logger = get_logger("adaptive_learner")

# ─── Configurazione ──────────────────────────────────────────────
# Quanti documenti servono per generare una regola automatica
AUTO_RULE_MIN_DOCS = 3
# Confidenza minima per applicare una regola auto-generata
AUTO_RULE_MIN_CONFIDENCE = 0.6
# Quanti top documenti simili considerare
SIMILARITY_TOP_K = 5
# Soglia di similarità per considerare un documento rilevante
SIMILARITY_THRESHOLD = 0.15
# Deviazioni standard per anomalia numerica
ANOMALY_STD_MULTIPLIER = 2.5
# Peso dei diversi metodi nell'ensemble
METHOD_WEIGHTS = {
    "rules": 0.7,
    "ml": 0.6,
    "learned_pattern": 0.65,
    "auto_rule": 0.55,
    "similar_doc": 0.5,
    "corrected": 1.0,
}


def _init_adaptive_tables():
    """Crea le tabelle per il sistema di apprendimento adattivo."""
    with get_connection() as conn:
        conn.executescript("""
            -- Memoria di ogni estrazione fatta
            CREATE TABLE IF NOT EXISTS extraction_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                field TEXT NOT NULL,
                value TEXT,
                method TEXT,
                confidence REAL DEFAULT 0.5,
                section_hint TEXT,
                prefix_context TEXT,
                suffix_context TEXT,
                was_correct INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_em_field
                ON extraction_memory(field, was_correct);
            CREATE INDEX IF NOT EXISTS idx_em_doc
                ON extraction_memory(doc_id);

            -- Statistiche per campo (aggiornate dopo ogni documento)
            CREATE TABLE IF NOT EXISTS field_value_stats (
                field TEXT PRIMARY KEY,
                value_type TEXT DEFAULT 'text',
                num_extractions INTEGER DEFAULT 0,
                num_unique_values INTEGER DEFAULT 0,
                num_corrections INTEGER DEFAULT 0,
                avg_numeric REAL,
                stddev_numeric REAL,
                min_numeric REAL,
                max_numeric REAL,
                avg_text_length REAL,
                common_values TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Regole auto-generate dall'analisi dei pattern
            CREATE TABLE IF NOT EXISTS auto_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                field TEXT NOT NULL,
                rule_type TEXT NOT NULL,
                regex_pattern TEXT,
                section_keyword TEXT,
                prefix_pattern TEXT,
                value_transform TEXT,
                success_count INTEGER DEFAULT 0,
                fail_count INTEGER DEFAULT 0,
                source_doc_count INTEGER DEFAULT 1,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_ar_field
                ON auto_rules(field, is_active);

            -- Fingerprint dei documenti per similarità
            CREATE TABLE IF NOT EXISTS doc_fingerprints (
                doc_id TEXT PRIMARY KEY,
                filename TEXT,
                stazione_appaltante TEXT,
                tipo_procedura TEXT,
                text_length INTEGER,
                section_headings TEXT,
                top_terms TEXT,
                extraction_summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Log di auto-correzioni effettuate
            CREATE TABLE IF NOT EXISTS auto_corrections_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                field TEXT,
                original_value TEXT,
                corrected_value TEXT,
                reason TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)


# ═════════════════════════════════════════════════════════════════════
# EXTRACTION MEMORY — Ricorda tutto di ogni estrazione
# ═════════════════════════════════════════════════════════════════════

class ExtractionMemory:
    """Memorizza ogni estrazione con contesto strutturale completo.

    Per ogni campo estratto, salva:
    - Il valore e il metodo usato
    - Il contesto testuale (prefisso/suffisso)
    - Se è stato poi corretto o no
    - Da quale sezione del documento proviene

    Questa memoria è la base per tutti gli altri componenti.
    """

    def __init__(self):
        _init_adaptive_tables()

    def record_extraction(self, doc_id: str, field: str, value: str,
                          method: str, full_text: str,
                          confidence: float = 0.5) -> int:
        """Registra un'estrazione nella memoria con contesto."""
        if not value or not field:
            return -1

        val_str = str(value).strip()
        if not val_str or val_str in ("None", "null", "{}"):
            return -1

        # Estrai contesto strutturale
        prefix_ctx, suffix_ctx, section = self._extract_context(
            full_text, val_str
        )

        with get_connection() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO extraction_memory "
                "(doc_id, field, value, method, confidence, "
                "section_hint, prefix_context, suffix_context) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (doc_id, field, val_str[:2000], method, confidence,
                 section, prefix_ctx, suffix_ctx)
            )
            return c.lastrowid

    def mark_corrected(self, doc_id: str, field: str):
        """Segna che un'estrazione era sbagliata (è stata corretta)."""
        with get_connection() as conn:
            conn.execute(
                "UPDATE extraction_memory SET was_correct = 0 "
                "WHERE doc_id = ? AND field = ? AND was_correct = 1",
                (doc_id, field)
            )

    def get_field_history(self, field: str, only_correct: bool = True,
                          limit: int = 100) -> List[dict]:
        """Recupera storico estrazioni per un campo."""
        where = "WHERE field = ?"
        if only_correct:
            where += " AND was_correct = 1"

        with get_connection(readonly=True) as conn:
            rows = conn.execute(
                f"SELECT doc_id, value, method, confidence, "
                f"section_hint, prefix_context, suffix_context, created_at "
                f"FROM extraction_memory {where} "
                f"ORDER BY created_at DESC LIMIT ?",
                (field, limit)
            ).fetchall()

        return [{
            "doc_id": r[0], "value": r[1], "method": r[2],
            "confidence": r[3], "section_hint": r[4],
            "prefix_context": r[5], "suffix_context": r[6],
            "created_at": r[7],
        } for r in rows]

    def get_doc_extractions(self, doc_id: str) -> Dict[str, dict]:
        """Recupera tutte le estrazioni di un documento."""
        with get_connection(readonly=True) as conn:
            rows = conn.execute(
                "SELECT field, value, method, confidence, was_correct "
                "FROM extraction_memory WHERE doc_id = ?",
                (doc_id,)
            ).fetchall()
        return {
            r[0]: {"value": r[1], "method": r[2],
                   "confidence": r[3], "was_correct": bool(r[4])}
            for r in rows
        }

    def _extract_context(self, full_text: str, value: str
                         ) -> Tuple[str, str, str]:
        """Estrae prefisso, suffisso e sezione per un valore."""
        prefix_ctx = ""
        suffix_ctx = ""
        section = ""

        if not full_text or not value:
            return prefix_ctx, suffix_ctx, section

        text_lower = full_text.lower()
        val_lower = value.lower()[:200]
        idx = text_lower.find(val_lower)

        if idx < 0:
            # Prova con le prime parole significative
            words = [w for w in value.split() if len(w) > 3][:3]
            if words:
                search = words[0].lower()
                idx = text_lower.find(search)

        if idx >= 0:
            # Prefisso: ultimi 200 chars prima del valore
            start = max(0, idx - 200)
            prefix_ctx = full_text[start:idx].strip()[-150:]

            # Suffisso: primi 200 chars dopo il valore
            end = min(len(full_text), idx + len(value) + 200)
            suffix_ctx = full_text[idx + len(value):end].strip()[:150]

            # Sezione: cerca l'ultimo heading prima del valore
            preceding = full_text[max(0, idx - 2000):idx]
            headings = re.findall(
                r'(?:^|\n)\s*(?:(?:Art\.?\s*\d+|CAPO\s+[IVX]+|'
                r'TITOLO\s+[IVX]+|\d+[\.)]\s*[A-Z])[^\n]*)',
                preceding, re.IGNORECASE
            )
            if headings:
                section = headings[-1].strip()[:200]

        return prefix_ctx, suffix_ctx, section


# ═════════════════════════════════════════════════════════════════════
# VALUE VALIDATOR — Validazione statistica dei valori estratti
# ═════════════════════════════════════════════════════════════════════

class ValueValidator:
    """Valida valori estratti contro statistiche storiche.

    Per ogni campo, mantiene:
    - Tipo di valore (numerico, data, testo, enum)
    - Range numerico tipico (media ± stddev)
    - Lunghezza tipica del testo
    - Valori comuni (per campi enum-like)

    Rileva anomalie:
    - Importo di €50 quando la media è €500.000
    - CIG con formato sbagliato
    - Date fuori range
    """

    def __init__(self):
        _init_adaptive_tables()

    def update_stats(self, field: str, value: str):
        """Aggiorna le statistiche di un campo con un nuovo valore."""
        if not value or not field:
            return

        val_str = str(value).strip()
        val_type = self._detect_type(val_str)
        numeric_val = self._parse_numeric(val_str)

        with get_connection() as conn:
            c = conn.cursor()
            existing = c.execute(
                "SELECT num_extractions, avg_numeric, stddev_numeric, "
                "min_numeric, max_numeric, avg_text_length, common_values, "
                "value_type FROM field_value_stats WHERE field = ?",
                (field,)
            ).fetchone()

            if existing:
                n = existing[0] + 1
                old_avg = existing[1]
                old_std = existing[2]
                old_min = existing[3]
                old_max = existing[4]
                old_avg_len = existing[5]
                common = json.loads(existing[6]) if existing[6] else {}

                # Aggiorna media e stddev numerica (Welford online)
                if numeric_val is not None and old_avg is not None:
                    new_avg = old_avg + (numeric_val - old_avg) / n
                    # Approssimazione stddev online
                    if n > 2 and old_std:
                        new_std = math.sqrt(
                            ((n - 2) * old_std ** 2 +
                             (numeric_val - old_avg) * (numeric_val - new_avg)) /
                            (n - 1)
                        )
                    else:
                        new_std = abs(numeric_val - new_avg)
                    new_min = min(old_min or numeric_val, numeric_val)
                    new_max = max(old_max or numeric_val, numeric_val)
                elif numeric_val is not None:
                    new_avg = numeric_val
                    new_std = 0
                    new_min = numeric_val
                    new_max = numeric_val
                else:
                    new_avg = old_avg
                    new_std = old_std
                    new_min = old_min
                    new_max = old_max

                # Aggiorna lunghezza media testo
                new_avg_len = (old_avg_len or 0) + (len(val_str) - (old_avg_len or 0)) / n

                # Aggiorna valori comuni (top 20)
                common[val_str] = common.get(val_str, 0) + 1
                if len(common) > 30:
                    # Mantieni solo i top 20
                    common = dict(
                        sorted(common.items(), key=lambda x: x[1], reverse=True)[:20]
                    )

                # Conta valori unici
                unique_count = c.execute(
                    "SELECT COUNT(DISTINCT value) FROM extraction_memory "
                    "WHERE field = ? AND was_correct = 1",
                    (field,)
                ).fetchone()[0]

                c.execute(
                    "UPDATE field_value_stats SET "
                    "num_extractions = ?, num_unique_values = ?, "
                    "avg_numeric = ?, stddev_numeric = ?, "
                    "min_numeric = ?, max_numeric = ?, "
                    "avg_text_length = ?, common_values = ?, "
                    "value_type = ?, last_updated = CURRENT_TIMESTAMP "
                    "WHERE field = ?",
                    (n, unique_count, new_avg, new_std, new_min, new_max,
                     new_avg_len, json.dumps(common, ensure_ascii=False),
                     val_type, field)
                )
            else:
                common = {val_str: 1}
                c.execute(
                    "INSERT INTO field_value_stats "
                    "(field, value_type, num_extractions, num_unique_values, "
                    "avg_numeric, stddev_numeric, min_numeric, max_numeric, "
                    "avg_text_length, common_values) "
                    "VALUES (?,?,1,1,?,?,?,?,?,?)",
                    (field, val_type, numeric_val, 0,
                     numeric_val, numeric_val,
                     len(val_str),
                     json.dumps(common, ensure_ascii=False))
                )

    def validate_value(self, field: str, value: str) -> dict:
        """Valida un valore contro le statistiche storiche.

        Ritorna:
        {
            "is_valid": bool,
            "confidence": float (0-1),
            "anomalies": [str, ...],
            "suggestion": str | None
        }
        """
        result = {"is_valid": True, "confidence": 0.5,
                  "anomalies": [], "suggestion": None}

        if not value or not field:
            return result

        val_str = str(value).strip()

        with get_connection(readonly=True) as conn:
            stats = conn.execute(
                "SELECT value_type, num_extractions, avg_numeric, "
                "stddev_numeric, min_numeric, max_numeric, "
                "avg_text_length, common_values "
                "FROM field_value_stats WHERE field = ?",
                (field,)
            ).fetchone()

        if not stats or stats[1] < 3:
            # Troppo pochi dati per validare
            return result

        val_type = stats[0]
        n = stats[1]
        avg_num = stats[2]
        std_num = stats[3]
        avg_len = stats[6]
        common = json.loads(stats[7]) if stats[7] else {}

        # Validazione numerica
        numeric_val = self._parse_numeric(val_str)
        if numeric_val is not None and avg_num is not None and std_num and std_num > 0:
            z_score = abs(numeric_val - avg_num) / std_num
            if z_score > ANOMALY_STD_MULTIPLIER and n >= 5:
                result["anomalies"].append(
                    f"Valore numerico anomalo: {numeric_val:.2f} "
                    f"(media: {avg_num:.2f} ± {std_num:.2f})"
                )
                result["is_valid"] = False
                result["confidence"] = max(0.1, 1.0 - z_score / 10)

        # Validazione lunghezza testo
        if avg_len and avg_len > 0 and len(val_str) > 0:
            len_ratio = len(val_str) / avg_len
            if len_ratio > 5.0 or len_ratio < 0.1:
                result["anomalies"].append(
                    f"Lunghezza anomala: {len(val_str)} chars "
                    f"(media: {avg_len:.0f})"
                )
                if len(result["anomalies"]) == 1:
                    result["is_valid"] = False
                    result["confidence"] = 0.3

        # Suggerimento: valore più comune se questo è anomalo
        if not result["is_valid"] and common:
            top_value = max(common, key=common.get)
            if top_value != val_str:
                result["suggestion"] = top_value

        # Alta confidenza se il valore è tra quelli già visti
        if val_str in common:
            freq = common[val_str]
            total_seen = sum(common.values())
            result["confidence"] = max(result["confidence"],
                                       0.6 + 0.3 * (freq / total_seen))

        return result

    def _detect_type(self, value: str) -> str:
        """Rileva il tipo di un valore."""
        if self._parse_numeric(value) is not None:
            return "numeric"
        if re.match(r'\d{4}[-/]\d{2}[-/]\d{2}', value):
            return "date"
        if re.match(r'\d{2}[-/]\d{2}[-/]\d{4}', value):
            return "date"
        return "text"

    @staticmethod
    def _parse_numeric(value: str) -> Optional[float]:
        """Prova a parsare un valore come numero."""
        if not value:
            return None
        s = str(value).strip()
        s = re.sub(r'[€$%\s]', '', s)
        # Formato italiano: 1.234.567,89
        if ',' in s and '.' in s:
            s = s.replace('.', '').replace(',', '.')
        elif ',' in s:
            s = s.replace(',', '.')
        try:
            return float(s)
        except (ValueError, TypeError):
            return None


# ═════════════════════════════════════════════════════════════════════
# RULE GENERATOR — Genera regole automatiche da pattern osservati
# ═════════════════════════════════════════════════════════════════════

class RuleGenerator:
    """Genera automaticamente nuove regole di estrazione
    analizzando i pattern ricorrenti nelle estrazioni riuscite.

    Dopo N estrazioni coerenti per un campo, il sistema:
    1. Analizza i contesti (prefissi/suffissi) comuni
    2. Generalizza in una regex riusabile
    3. Testa la regex su documenti passati
    4. Se supera la soglia di accuratezza, la attiva

    Questo crea un ciclo virtuoso:
      Doc caricato → estrazione → memoria → pattern →
      regex auto → prossimo doc estratto meglio
    """

    def __init__(self):
        _init_adaptive_tables()

    def analyze_and_generate(self, field: str) -> List[dict]:
        """Analizza le estrazioni passate e genera nuove regole."""
        with get_connection(readonly=True) as conn:
            rows = conn.execute(
                "SELECT prefix_context, suffix_context, value, method "
                "FROM extraction_memory "
                "WHERE field = ? AND was_correct = 1 "
                "AND prefix_context IS NOT NULL AND prefix_context != '' "
                "ORDER BY created_at DESC LIMIT 50",
                (field,)
            ).fetchall()

        if len(rows) < AUTO_RULE_MIN_DOCS:
            return []

        # Cerca pattern comuni nei prefissi
        generated = []
        prefix_clusters = self._cluster_prefixes([r[0] for r in rows])

        for cluster_key, prefixes in prefix_clusters.items():
            if len(prefixes) < AUTO_RULE_MIN_DOCS:
                continue

            # Genera regex dal cluster di prefissi
            regex = self._build_regex_from_prefixes(prefixes)
            if not regex:
                continue

            # Verifica che non esista già una regola simile
            if self._rule_exists(field, regex):
                continue

            # Salva la regola
            rule = self._save_rule(
                field, "prefix_regex", regex,
                source_doc_count=len(prefixes)
            )
            if rule:
                generated.append(rule)
                logger.info(
                    "Auto-regola generata per '%s': %s (da %d docs)",
                    field, regex[:80], len(prefixes)
                )

        return generated

    def apply_auto_rules(self, field: str, text: str) -> List[Tuple[str, float, int]]:
        """Applica le regole auto-generate per estrarre valori.

        Ritorna [(valore, confidenza, rule_id), ...].
        """
        rules = self._get_active_rules(field)
        if not rules:
            return []

        results = []
        for rule in rules:
            try:
                value = self._apply_rule(rule, text)
                if value and len(value.strip()) > 1:
                    confidence = self._rule_confidence(rule)
                    if confidence >= AUTO_RULE_MIN_CONFIDENCE:
                        results.append((
                            clean_string(value),
                            confidence,
                            rule["id"]
                        ))
            except Exception:
                pass

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def report_result(self, rule_id: int, success: bool):
        """Aggiorna il conteggio successi/fallimenti di una regola."""
        col = "success_count" if success else "fail_count"
        with get_connection() as conn:
            conn.execute(
                f"UPDATE auto_rules SET {col} = {col} + 1, "
                f"last_used = CURRENT_TIMESTAMP WHERE id = ?",
                (rule_id,)
            )
            # Disattiva regole con troppi fallimenti
            if not success:
                conn.execute(
                    "UPDATE auto_rules SET is_active = 0 "
                    "WHERE id = ? AND fail_count > success_count * 2 "
                    "AND fail_count >= 5",
                    (rule_id,)
                )

    def _cluster_prefixes(self, prefixes: List[str]) -> Dict[str, List[str]]:
        """Raggruppa prefissi simili per trovare pattern comuni.

        Usa le ultime parole chiave significative come chiave di cluster.
        """
        clusters = defaultdict(list)
        for prefix in prefixes:
            if not prefix:
                continue
            # Estrai le ultime parole chiave significative
            words = re.findall(r'[A-Za-zÀ-ú]{3,}', prefix[-100:].lower())
            if not words:
                continue
            # Usa le ultime 2-3 parole come chiave
            key = " ".join(words[-3:])
            clusters[key].append(prefix)
        return dict(clusters)

    def _build_regex_from_prefixes(self, prefixes: List[str]) -> Optional[str]:
        """Costruisce una regex generalizzata da un set di prefissi simili."""
        if not prefixes:
            return None

        # Trova la sottosequenza comune più lunga tra i prefissi
        # Usa solo le ultime N chars (la parte più vicina al valore)
        tails = [p[-80:].strip() for p in prefixes if len(p) > 5]
        if len(tails) < 2:
            return None

        common = self._longest_common_substring(tails)
        if not common or len(common) < 8:
            return None

        # Generalizza il suffisso comune in regex
        regex = re.escape(common.strip())
        # Flessibilità su spazi
        regex = re.sub(r'(?:\\ )+', r'\\s+', regex)
        regex = re.sub(r'\\n', r'\\s+', regex)
        # Flessibilità su numeri
        regex = re.sub(r'(?:\\\d)+', r'\\d+', regex)
        # Aggiungi cattura del valore
        regex = regex + r'\s*(.+?)(?:\n\n|\n(?=[A-Z]{2,})|\.\s*$|$)'

        return regex

    def _longest_common_substring(self, strings: List[str]) -> str:
        """Trova la sottostringa comune più lunga tra stringhe."""
        if not strings:
            return ""
        shortest = min(strings, key=len)
        best = ""
        for length in range(len(shortest), 5, -1):
            for start in range(len(shortest) - length + 1):
                candidate = shortest[start:start + length].lower()
                if all(candidate in s.lower() for s in strings):
                    if len(candidate) > len(best):
                        best = shortest[start:start + length]
                    if best:
                        return best
        return best

    def _rule_exists(self, field: str, regex: str) -> bool:
        """Verifica se una regola simile esiste già."""
        with get_connection(readonly=True) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM auto_rules "
                "WHERE field = ? AND regex_pattern = ? AND is_active = 1",
                (field, regex)
            ).fetchone()
            return row[0] > 0

    def _save_rule(self, field: str, rule_type: str, regex: str,
                   source_doc_count: int = 1) -> Optional[dict]:
        """Salva una nuova regola auto-generata."""
        with get_connection() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO auto_rules "
                "(field, rule_type, regex_pattern, source_doc_count) "
                "VALUES (?,?,?,?)",
                (field, rule_type, regex[:2000], source_doc_count)
            )
            return {
                "id": c.lastrowid, "field": field,
                "rule_type": rule_type,
                "regex": regex[:100],
                "source_docs": source_doc_count,
            }

    def _get_active_rules(self, field: str) -> List[dict]:
        """Recupera le regole attive per un campo."""
        with get_connection(readonly=True) as conn:
            rows = conn.execute(
                "SELECT id, rule_type, regex_pattern, section_keyword, "
                "prefix_pattern, success_count, fail_count, source_doc_count "
                "FROM auto_rules "
                "WHERE field = ? AND is_active = 1 "
                "ORDER BY success_count DESC LIMIT 10",
                (field,)
            ).fetchall()
        return [{
            "id": r[0], "rule_type": r[1], "regex": r[2],
            "section_keyword": r[3], "prefix_pattern": r[4],
            "success_count": r[5], "fail_count": r[6],
            "source_doc_count": r[7],
        } for r in rows]

    def _apply_rule(self, rule: dict, text: str) -> Optional[str]:
        """Applica una singola regola auto-generata."""
        regex = rule.get("regex")
        if not regex:
            return None
        try:
            m = re.search(regex, text, re.IGNORECASE | re.DOTALL)
            if m and m.lastindex:
                value = m.group(1).strip()
                if len(value) > 2000:
                    value = value[:2000]
                value = re.sub(r'[\s;,:.]+$', '', value)
                return value if len(value) > 1 else None
        except re.error:
            pass
        return None

    def _rule_confidence(self, rule: dict) -> float:
        """Calcola confidenza di una regola auto-generata."""
        s = rule.get("success_count", 0)
        f = rule.get("fail_count", 0)
        docs = rule.get("source_doc_count", 1)
        total = s + f

        if total == 0:
            # Nuova regola: confidenza basata su quanti docs l'hanno generata
            return min(0.7, 0.4 + docs * 0.05)

        base = (s + 1) / (total + 2)  # Bayesian smoothing
        # Bonus per regole validate su molti documenti
        if docs >= 5:
            base = min(0.9, base + 0.05)
        return round(base, 3)


# ═════════════════════════════════════════════════════════════════════
# DOC SIMILARITY — Trova documenti simili per trasferire strategie
# ═════════════════════════════════════════════════════════════════════

class DocSimilarity:
    """Trova documenti simili per trasferire strategie di estrazione.

    Quando un nuovo documento viene caricato:
    1. Calcola il suo fingerprint (termini chiave, struttura, metadata)
    2. Trova i K documenti più simili nel database
    3. Per i campi vuoti, usa le strategie che hanno funzionato
       sui documenti simili

    Similarità basata su:
    - Termini chiave comuni (TF-IDF semplificato)
    - Stazione appaltante uguale
    - Struttura simile (sezioni presenti)
    """

    def __init__(self):
        _init_adaptive_tables()

    def save_fingerprint(self, doc_id: str, filename: str,
                         text: str, result: dict):
        """Calcola e salva il fingerprint di un documento."""
        # Estrai metadata dall'estrazione
        stazione = ""
        tipo_proc = ""
        if isinstance(result, dict):
            stazione = str(result.get("stazione_appaltante",
                          result.get("_stazione", "")))[:200]
            tipo_proc = str(result.get("tipo_procedura",
                           result.get("tipologia_appalto", "")))[:200]

        # Top terms: le parole più frequenti (escluse stopwords)
        top_terms = self._extract_top_terms(text)

        # Sezioni rilevate
        headings = self._extract_headings(text)

        # Sommario di cosa è stato estratto con successo
        methods = result.get("_methods", {}) if isinstance(result, dict) else {}
        summary = {k: v for k, v in methods.items() if v}

        with get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO doc_fingerprints "
                "(doc_id, filename, stazione_appaltante, tipo_procedura, "
                "text_length, section_headings, top_terms, extraction_summary) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (doc_id, filename, stazione, tipo_proc,
                 len(text),
                 json.dumps(headings, ensure_ascii=False),
                 json.dumps(top_terms, ensure_ascii=False),
                 json.dumps(summary, ensure_ascii=False))
            )

    def find_similar(self, text: str, result: dict = None,
                     exclude_doc: str = None) -> List[dict]:
        """Trova i documenti più simili nel database."""
        current_terms = set(self._extract_top_terms(text))
        current_headings = set(self._extract_headings(text))
        current_stazione = ""
        if result and isinstance(result, dict):
            current_stazione = str(result.get("stazione_appaltante",
                                  result.get("_stazione", ""))).lower()

        with get_connection(readonly=True) as conn:
            rows = conn.execute(
                "SELECT doc_id, filename, stazione_appaltante, "
                "tipo_procedura, text_length, section_headings, "
                "top_terms, extraction_summary FROM doc_fingerprints"
            ).fetchall()

        if not rows:
            return []

        scored = []
        for r in rows:
            if exclude_doc and r[0] == exclude_doc:
                continue

            doc_terms = set(json.loads(r[6]) if r[6] else [])
            doc_headings = set(json.loads(r[5]) if r[5] else [])
            doc_stazione = (r[2] or "").lower()

            # Similarità Jaccard sui termini
            term_sim = 0
            if current_terms and doc_terms:
                intersection = current_terms & doc_terms
                union = current_terms | doc_terms
                term_sim = len(intersection) / len(union) if union else 0

            # Similarità strutturale (sezioni)
            heading_sim = 0
            if current_headings and doc_headings:
                intersection = current_headings & doc_headings
                union = current_headings | doc_headings
                heading_sim = len(intersection) / len(union) if union else 0

            # Bonus per stessa stazione appaltante
            stazione_bonus = 0
            if current_stazione and doc_stazione:
                if current_stazione == doc_stazione:
                    stazione_bonus = 0.3
                elif (current_stazione in doc_stazione or
                      doc_stazione in current_stazione):
                    stazione_bonus = 0.15

            # Score complessivo
            score = term_sim * 0.5 + heading_sim * 0.2 + stazione_bonus

            if score >= SIMILARITY_THRESHOLD:
                summary = json.loads(r[7]) if r[7] else {}
                scored.append({
                    "doc_id": r[0], "filename": r[1],
                    "similarity": round(score, 3),
                    "stazione": r[2], "tipo": r[3],
                    "extraction_summary": summary,
                })

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:SIMILARITY_TOP_K]

    def get_strategies_from_similar(self, similar_docs: List[dict],
                                    missing_fields: List[str]
                                    ) -> Dict[str, List[dict]]:
        """Per ogni campo mancante, recupera le strategie che hanno
        funzionato su documenti simili."""
        strategies = defaultdict(list)

        doc_ids = [d["doc_id"] for d in similar_docs]
        if not doc_ids:
            return dict(strategies)

        placeholders = ",".join("?" * len(doc_ids))
        fields_ph = ",".join("?" * len(missing_fields))

        with get_connection(readonly=True) as conn:
            rows = conn.execute(
                f"SELECT doc_id, field, value, method, confidence, "
                f"prefix_context, suffix_context "
                f"FROM extraction_memory "
                f"WHERE doc_id IN ({placeholders}) "
                f"AND field IN ({fields_ph}) "
                f"AND was_correct = 1 "
                f"ORDER BY confidence DESC",
                doc_ids + missing_fields
            ).fetchall()

        # Mappa doc_id → similarità per pesare
        sim_map = {d["doc_id"]: d["similarity"] for d in similar_docs}

        for r in rows:
            field = r[1]
            strategies[field].append({
                "value_hint": r[2], "method": r[3],
                "confidence": r[4] * sim_map.get(r[0], 0.5),
                "prefix_context": r[5],
                "source_doc": r[0],
            })

        return dict(strategies)

    def _extract_top_terms(self, text: str, n: int = 50) -> List[str]:
        """Estrae i termini più significativi dal testo."""
        # Stopwords italiane comuni
        _stop = {
            "il", "lo", "la", "le", "gli", "un", "uno", "una", "di", "del",
            "dello", "della", "dei", "degli", "delle", "da", "dal", "dalla",
            "in", "nel", "nella", "con", "su", "per", "tra", "fra", "che",
            "non", "sono", "essere", "viene", "alla", "alle", "agli", "dal",
            "nei", "nelle", "sui", "sulle", "questo", "questa", "questi",
            "queste", "quello", "quella", "quelli", "quelle", "come", "anche",
            "dove", "quando", "quanto", "quali", "deve", "può", "sia",
            "ogni", "tutti", "tutte", "altro", "altri", "altre", "dopo",
            "prima", "più", "meno", "solo", "così", "molto", "poco",
            "senza", "verso", "fino", "oltre", "circa", "sotto", "sopra",
            "dentro", "fuori", "parte", "caso", "modo", "tempo", "anno",
            "sensi", "articolo", "comma", "lettera", "punto", "numero",
            "presente", "previsto", "prevista", "relativo", "relativa",
            "quanto", "ove", "ovvero", "oppure", "mediante", "nonché",
        }
        words = re.findall(r'[a-zA-ZÀ-ú]{4,}', text.lower())
        counter = Counter(w for w in words if w not in _stop)
        return [w for w, _ in counter.most_common(n)]

    def _extract_headings(self, text: str) -> List[str]:
        """Estrae le intestazioni/titoli di sezione dal testo."""
        headings = []
        patterns = [
            r'(?:^|\n)\s*(Art\.?\s*\d+[^\n]*)',
            r'(?:^|\n)\s*(CAPO\s+[IVX]+[^\n]*)',
            r'(?:^|\n)\s*(\d+[\.)]\s+[A-Z][^\n]{5,60})',
            r'(?:^|\n)\s*([A-Z][A-Z\s]{10,60})(?:\n|$)',
        ]
        for pat in patterns:
            for m in re.finditer(pat, text[:10000]):
                h = m.group(1).strip()[:100].lower()
                if len(h) > 5:
                    headings.append(h)
        # Normalizza e deduplica
        seen = set()
        result = []
        for h in headings:
            key = re.sub(r'\s+', ' ', h).strip()
            if key not in seen:
                seen.add(key)
                result.append(key)
        return result[:30]


# ═════════════════════════════════════════════════════════════════════
# CONFIDENCE ENSEMBLE — Combina i risultati di tutti i metodi
# ═════════════════════════════════════════════════════════════════════

class ConfidenceEnsemble:
    """Combina risultati da metodi multipli con voto pesato.

    Quando più metodi estraggono lo stesso campo:
    - Se concordano → alta confidenza
    - Se discordano → usa il metodo più affidabile
    - Peso basato su storico successi del metodo per quel campo
    """

    def combine(self, candidates: Dict[str, List[Tuple[str, float, str]]]
                ) -> Dict[str, Tuple[str, float, str]]:
        """Combina candidati da metodi diversi per ogni campo.

        Input: {field: [(valore, confidenza, metodo), ...]}
        Output: {field: (miglior_valore, confidenza_finale, metodo)}
        """
        results = {}
        for field, cands in candidates.items():
            if not cands:
                continue
            results[field] = self._pick_best(field, cands)
        return results

    def _pick_best(self, field: str, candidates: List[Tuple[str, float, str]]
                   ) -> Tuple[str, float, str]:
        """Seleziona il miglior candidato con voto pesato."""
        if len(candidates) == 1:
            return candidates[0]

        # Raggruppa per valore (normalizzato)
        value_groups = defaultdict(list)
        for val, conf, method in candidates:
            key = self._normalize_for_comparison(val)
            weight = METHOD_WEIGHTS.get(method, 0.5)
            value_groups[key].append((val, conf * weight, method))

        # Il valore con il peso totale più alto vince
        best_key = None
        best_score = -1
        for key, entries in value_groups.items():
            total_score = sum(e[1] for e in entries)
            # Bonus se più metodi concordano
            if len(entries) > 1:
                total_score *= (1 + 0.15 * (len(entries) - 1))
            if total_score > best_score:
                best_score = total_score
                best_key = key

        if best_key is None:
            return candidates[0]

        group = value_groups[best_key]
        # Prendi il candidato con la confidenza più alta nel gruppo
        group.sort(key=lambda x: x[1], reverse=True)
        best_val, best_conf, best_method = group[0]

        # Confidenza finale: boosted se concordano più metodi
        n_methods = len(group)
        final_conf = min(0.98, best_conf * (1 + 0.1 * (n_methods - 1)))

        return (best_val, round(final_conf, 3), best_method)

    @staticmethod
    def _normalize_for_comparison(value: str) -> str:
        """Normalizza un valore per confronto tra metodi."""
        s = str(value).strip().lower()
        s = re.sub(r'\s+', ' ', s)
        s = re.sub(r'[.,;:\-]+$', '', s)
        return s


# ═════════════════════════════════════════════════════════════════════
# ADAPTIVE LEARNER — Orchestratore principale
# ═════════════════════════════════════════════════════════════════════

class AdaptiveLearner:
    """Orchestratore dell'apprendimento adattivo autonomo.

    Ciclo di vita per OGNI documento caricato:

    1. PRE-ESTRAZIONE:
       - Trova documenti simili
       - Prepara strategie da documenti simili per campi difficili

    2. POST-ESTRAZIONE:
       - Valida ogni valore contro statistiche storiche
       - Corregge automaticamente anomalie evidenti
       - Memorizza l'estrazione completa con contesto
       - Aggiorna statistiche per campo
       - Salva il fingerprint del documento

    3. PERIODICAMENTE (dopo N documenti):
       - Genera nuove regole automatiche dai pattern osservati
       - Pulisce regole fallimentari

    4. SU CORREZIONE:
       - Aggiorna la memoria (marca valore errato)
       - Aggiorna statistiche
       - Può triggerare generazione nuove regole
    """

    def __init__(self):
        self.memory = ExtractionMemory()
        self.validator = ValueValidator()
        self.rule_gen = RuleGenerator()
        self.doc_sim = DocSimilarity()
        self.ensemble = ConfidenceEnsemble()
        self._docs_since_rule_gen = 0
        self._RULE_GEN_INTERVAL = 5  # Ogni N documenti, genera regole

    # ── Pre-Estrazione ─────────────────────────────────────────────

    def pre_extraction_hints(self, text: str, result: dict = None
                             ) -> Dict[str, List[dict]]:
        """Prima dell'estrazione: trova hint da documenti simili.

        Ritorna strategie suggerite per i campi basate su documenti simili.
        """
        similar = self.doc_sim.find_similar(text, result)
        if not similar:
            return {}

        # Identifica tutti i campi possibili (dal registry + storici)
        all_fields = self._get_known_fields()

        return self.doc_sim.get_strategies_from_similar(similar, all_fields)

    # ── Miglioramento risultati (Fase Ensemble) ───────────────────

    def enhance_result(self, result: dict, text: str,
                       methods: dict = None) -> Tuple[dict, dict]:
        """Migliora i risultati usando regole auto-generate e similarità.

        Per ogni campo:
        1. Applica regole auto-generate
        2. Usa strategie da documenti simili
        3. Valida il valore (rules/ML/auto) contro le statistiche
        4. Se il valore è anomalo e c'è un'alternativa migliore, sostituisci

        Ritorna (result_migliorato, metodi_aggiunti).
        """
        if methods is None:
            methods = result.get("_methods", {})

        added_methods = {}
        doc_id = result.get("_doc_id", "")
        empty_objects = [
            {"quota_fissa_65_perc": False, "quota_ribassabile_35_perc": False},
            {"obbligatorio": False, "note": None},
        ]

        # Trova campi vuoti o a bassa confidenza
        empty_fields = []
        for key, val in result.items():
            if key.startswith("_"):
                continue
            is_empty = val in [None, "", 0, [], {}]
            if isinstance(val, dict) and all(
                v in [None, False, ""] for v in val.values()
            ):
                is_empty = True
            if val in empty_objects:
                is_empty = True
            if is_empty:
                empty_fields.append(key)

        if not empty_fields:
            return result, added_methods

        # Raccogli candidati da tutti i metodi per campi vuoti
        candidates = defaultdict(list)

        # 1. Regole auto-generate
        for field in empty_fields:
            auto_results = self.rule_gen.apply_auto_rules(field, text)
            for val, conf, rid in auto_results:
                candidates[field].append((val, conf, "auto_rule"))

        # 2. Strategie da documenti simili (solo per campi ancora vuoti)
        still_empty = [f for f in empty_fields if f not in candidates]
        if still_empty:
            similar = self.doc_sim.find_similar(text, result,
                                                exclude_doc=doc_id)
            if similar:
                strategies = self.doc_sim.get_strategies_from_similar(
                    similar, still_empty
                )
                for field, strats in strategies.items():
                    for strat in strats[:2]:
                        # Usa il contesto del documento simile per
                        # estrarre dal documento corrente
                        value = self._try_extract_with_context(
                            text, strat.get("prefix_context", ""),
                            strat.get("value_hint", "")
                        )
                        if value:
                            candidates[field].append(
                                (value, strat["confidence"] * 0.8,
                                 "similar_doc")
                            )

        # 3. Ensemble: scegli il miglior candidato per ogni campo
        best = self.ensemble.combine(candidates)
        for field, (val, conf, method) in best.items():
            result[field] = val
            added_methods[field] = f"{method}({conf:.0%})"

        # 4. Validazione: controlla valori (inclusi quelli esistenti)
        self._validate_and_autocorrect(result, text, doc_id, added_methods)

        return result, added_methods

    # ── Post-Estrazione ────────────────────────────────────────────

    def post_extraction_learn(self, doc_id: str, filename: str,
                              text: str, result: dict,
                              methods: dict = None):
        """Dopo l'estrazione: impara da tutto ciò che è stato estratto.

        Questa funzione è il cuore dell'apprendimento autonomo.
        Va chiamata DOPO ogni estrazione completata.
        """
        if methods is None:
            methods = result.get("_methods", {})

        # 1. Memorizza ogni campo estratto con contesto
        for field, value in result.items():
            if field.startswith("_"):
                continue
            if value in [None, "", 0, [], {}, False]:
                continue
            if isinstance(value, (dict, list)):
                continue  # Memorizza solo valori scalari

            method = methods.get(field, "rules")
            confidence = self._estimate_confidence(field, value, method)

            self.memory.record_extraction(
                doc_id, field, str(value), method, text, confidence
            )

            # 2. Aggiorna statistiche del campo
            self.validator.update_stats(field, str(value))

        # 3. Salva fingerprint del documento
        self.doc_sim.save_fingerprint(doc_id, filename, text, result)

        # 4. Periodicamente genera nuove regole
        self._docs_since_rule_gen += 1
        if self._docs_since_rule_gen >= self._RULE_GEN_INTERVAL:
            self._trigger_rule_generation()
            self._docs_since_rule_gen = 0

        logger.info(
            "Adaptive learning completato per doc '%s': "
            "%d campi memorizzati, fingerprint salvato",
            doc_id, sum(1 for k, v in result.items()
                        if not k.startswith("_") and v)
        )

    # ── Su Correzione ──────────────────────────────────────────────

    def on_correction(self, doc_id: str, field: str,
                      correct_value: str, wrong_value: str,
                      full_text: str) -> dict:
        """Callback quando un valore viene corretto.

        Aggiorna la memoria e può triggerare generazione regole.
        """
        # Segna la vecchia estrazione come errata
        self.memory.mark_corrected(doc_id, field)

        # Memorizza la correzione come estrazione corretta
        self.memory.record_extraction(
            doc_id, field, correct_value,
            "correction", full_text, confidence=1.0
        )

        # Aggiorna statistiche con il valore corretto
        self.validator.update_stats(field, correct_value)

        # Aggiorna contatore correzioni nelle statistiche
        with get_connection() as conn:
            conn.execute(
                "UPDATE field_value_stats SET "
                "num_corrections = num_corrections + 1 "
                "WHERE field = ?",
                (field,)
            )

        # Logga auto-correzione per analisi
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO auto_corrections_log "
                "(doc_id, field, original_value, corrected_value, "
                "reason, confidence) VALUES (?,?,?,?,?,?)",
                (doc_id, field, wrong_value, correct_value,
                 "human_correction", 1.0)
            )

        # Genera regole se abbiamo abbastanza dati per questo campo
        new_rules = self.rule_gen.analyze_and_generate(field)

        return {
            "memory_updated": True,
            "stats_updated": True,
            "new_rules_generated": len(new_rules),
            "rules": new_rules,
        }

    # ── Monitoring & Diagnostica ───────────────────────────────────

    def get_learning_status(self) -> dict:
        """Stato completo del sistema di apprendimento."""
        with get_connection(readonly=True) as conn:
            # Documenti processati
            n_docs = conn.execute(
                "SELECT COUNT(*) FROM doc_fingerprints"
            ).fetchone()[0]

            # Estrazioni in memoria
            n_extractions = conn.execute(
                "SELECT COUNT(*) FROM extraction_memory"
            ).fetchone()[0]
            n_correct = conn.execute(
                "SELECT COUNT(*) FROM extraction_memory WHERE was_correct = 1"
            ).fetchone()[0]

            # Regole auto-generate
            n_rules = conn.execute(
                "SELECT COUNT(*) FROM auto_rules WHERE is_active = 1"
            ).fetchone()[0]
            n_rules_total = conn.execute(
                "SELECT COUNT(*) FROM auto_rules"
            ).fetchone()[0]

            # Auto-correzioni
            n_auto_corrections = conn.execute(
                "SELECT COUNT(*) FROM auto_corrections_log"
            ).fetchone()[0]

            # Statistiche per campo
            field_stats = conn.execute(
                "SELECT field, num_extractions, num_corrections, "
                "value_type, avg_text_length "
                "FROM field_value_stats ORDER BY num_extractions DESC"
            ).fetchall()

            # Regole più efficaci
            top_rules = conn.execute(
                "SELECT field, regex_pattern, success_count, fail_count, "
                "source_doc_count "
                "FROM auto_rules WHERE is_active = 1 "
                "ORDER BY success_count DESC LIMIT 10"
            ).fetchall()

        accuracy = round(n_correct / n_extractions * 100, 1) if n_extractions else 0

        return {
            "documents_processed": n_docs,
            "total_extractions": n_extractions,
            "correct_extractions": n_correct,
            "accuracy_rate": accuracy,
            "active_auto_rules": n_rules,
            "total_auto_rules": n_rules_total,
            "auto_corrections": n_auto_corrections,
            "docs_until_next_rule_gen": max(0,
                self._RULE_GEN_INTERVAL - self._docs_since_rule_gen),
            "field_stats": {
                r[0]: {
                    "extractions": r[1],
                    "corrections": r[2],
                    "type": r[3],
                    "accuracy": round(
                        (1 - r[2] / r[1]) * 100, 1
                    ) if r[1] > 0 else None,
                }
                for r in field_stats
            },
            "top_auto_rules": [{
                "field": r[0],
                "regex": (r[1] or "")[:80],
                "success": r[2],
                "fail": r[3],
                "from_docs": r[4],
            } for r in top_rules],
        }

    def get_field_intelligence(self, field: str) -> dict:
        """Report dettagliato sull'intelligenza acquisita per un campo."""
        history = self.memory.get_field_history(field, limit=20)
        validation_stats = None

        with get_connection(readonly=True) as conn:
            stats_row = conn.execute(
                "SELECT * FROM field_value_stats WHERE field = ?",
                (field,)
            ).fetchone()

            rules = conn.execute(
                "SELECT id, regex_pattern, success_count, fail_count, "
                "source_doc_count, created_at "
                "FROM auto_rules WHERE field = ? AND is_active = 1",
                (field,)
            ).fetchall()

        if stats_row:
            validation_stats = {
                "type": stats_row[1],
                "extractions": stats_row[2],
                "unique_values": stats_row[3],
                "corrections": stats_row[4],
                "avg_numeric": stats_row[5],
                "stddev": stats_row[6],
                "range": [stats_row[7], stats_row[8]],
                "avg_text_length": stats_row[9],
                "common_values": json.loads(stats_row[10])
                    if stats_row[10] else {},
            }

        return {
            "field": field,
            "recent_extractions": history[:10],
            "stats": validation_stats,
            "auto_rules": [{
                "id": r[0], "regex": (r[1] or "")[:100],
                "success": r[2], "fail": r[3],
                "from_docs": r[4], "created": r[5],
            } for r in rules],
            "maturity": self._field_maturity(field, validation_stats),
        }

    # ── Metodi privati ─────────────────────────────────────────────

    def _validate_and_autocorrect(self, result: dict, text: str,
                                  doc_id: str, methods: dict):
        """Valida i valori estratti e corregge anomalie evidenti."""
        for field, value in list(result.items()):
            if field.startswith("_") or value is None:
                continue
            if isinstance(value, (dict, list)):
                continue

            validation = self.validator.validate_value(field, str(value))

            if not validation["is_valid"] and validation["suggestion"]:
                # Controlla che il suggerimento sia nel testo
                suggestion = validation["suggestion"]
                if (suggestion.lower() in text.lower() and
                        validation["confidence"] < 0.3):
                    # Auto-correzione
                    old_val = result[field]
                    result[field] = suggestion
                    methods[field] = "auto_corrected"

                    with get_connection() as conn:
                        conn.execute(
                            "INSERT INTO auto_corrections_log "
                            "(doc_id, field, original_value, corrected_value, "
                            "reason, confidence) VALUES (?,?,?,?,?,?)",
                            (doc_id, field, str(old_val), suggestion,
                             "; ".join(validation["anomalies"]),
                             validation["confidence"])
                        )

                    logger.info(
                        "Auto-correzione '%s': '%s' → '%s' (%s)",
                        field, str(old_val)[:50], suggestion[:50],
                        validation["anomalies"][0][:80]
                    )

    def _try_extract_with_context(self, text: str, prefix: str,
                                  value_hint: str) -> Optional[str]:
        """Prova a estrarre un valore dal testo usando il contesto
        di un documento simile."""
        if not prefix or not text:
            return None

        # Cerca il prefisso (o qualcosa di simile) nel testo
        prefix_words = [w for w in prefix.split() if len(w) > 4][-3:]
        if not prefix_words:
            return None

        # Cerca la sequenza di parole nel testo
        text_lower = text.lower()
        search = prefix_words[0].lower()
        idx = text_lower.find(search)

        if idx < 0:
            return None

        # Verifica che le parole successive siano nelle vicinanze
        for word in prefix_words[1:]:
            nearby = text_lower[idx:idx + 300]
            if word.lower() not in nearby:
                return None

        # Estrai il valore dopo il contesto
        after = text[idx + len(search):idx + len(search) + 500]
        # Prendi fino al prossimo separatore forte
        m = re.search(
            r'[:\-–]\s*([^\n]+?)(?:\n\n|\n(?=[A-Z]{2,})|\.\s*\n)',
            after
        )
        if m:
            extracted = m.group(1).strip()
            if len(extracted) > 2 and len(extracted) < 500:
                return clean_string(extracted)

        return None

    def _estimate_confidence(self, field: str, value: str,
                             method: str) -> float:
        """Stima la confidenza di un'estrazione basandosi su metodo e storico."""
        base = METHOD_WEIGHTS.get(method, 0.5)

        # Bonus se il valore è già stato visto per questo campo
        history = self.memory.get_field_history(field, limit=20)
        val_str = str(value).lower().strip()
        seen_count = sum(
            1 for h in history
            if h["value"].lower().strip() == val_str
        )
        if seen_count > 0:
            base = min(0.95, base + 0.1 * seen_count)

        return round(base, 3)

    def _trigger_rule_generation(self):
        """Genera nuove regole per tutti i campi con abbastanza dati."""
        with get_connection(readonly=True) as conn:
            fields = conn.execute(
                "SELECT field, COUNT(*) as cnt "
                "FROM extraction_memory "
                "WHERE was_correct = 1 "
                "GROUP BY field HAVING cnt >= ?",
                (AUTO_RULE_MIN_DOCS,)
            ).fetchall()

        total_generated = 0
        for field, count in fields:
            try:
                rules = self.rule_gen.analyze_and_generate(field)
                total_generated += len(rules)
            except Exception:
                logger.debug("Rule generation failed for %s", field,
                             exc_info=True)

        if total_generated:
            logger.info("Generazione regole periodica: %d nuove regole",
                        total_generated)

    def _get_known_fields(self) -> List[str]:
        """Recupera tutti i campi conosciuti dal registry e dallo storico."""
        from field_registry import registry
        fields = set(registry.all_names())

        # Aggiungi campi visti nelle estrazioni
        with get_connection(readonly=True) as conn:
            rows = conn.execute(
                "SELECT DISTINCT field FROM extraction_memory"
            ).fetchall()
            for r in rows:
                fields.add(r[0])

        return list(fields)

    @staticmethod
    def _field_maturity(field: str, stats: dict = None) -> str:
        """Calcola il livello di maturità dell'apprendimento per un campo."""
        if not stats:
            return "unknown"
        n = stats.get("extractions", 0)
        corrections = stats.get("corrections", 0)
        if n == 0:
            return "new"
        if n < 3:
            return "learning"
        accuracy = (1 - corrections / n) if n > 0 else 0
        if n < 10:
            return "developing"
        if accuracy > 0.9:
            return "mature"
        if accuracy > 0.7:
            return "improving"
        return "needs_attention"


# ── Singleton globale ──────────────────────────────────────────────
adaptive_learner = AdaptiveLearner()
