"""
AppaltoAI — Smart Learner (Apprendimento Progressivo Autonomo)
================================================================
Risolve il problema fondamentale del ML classico nel sistema:
  LogisticRegression predice VALORI già visti → inutile per campi unici.

Questo modulo aggiunge:
  1. PatternLearner: impara DOVE trovare i valori (pattern strutturali)
  2. AutoTrainer: ri-addestra automaticamente dopo N correzioni
  3. ExtractionMemory: memorizza pattern vincenti per ogni campo
  4. SelfEvaluator: monitora degradazione e attiva auto-correzione

Principio chiave:
  Per "oggetto_appalto", ogni PDF ha un testo diverso.
  Il ML classico non può generalizzare perché predice tra valori GIÀ VISTI.
  Il PatternLearner impara i PATTERN STRUTTURALI: "il valore si trova
  dopo 'OGGETTO:' e prima di '\\n\\nCIG'" — questo generalizza a PDF nuovi.
"""

import json
import re
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import Counter

from config import (
    DB_PATH, MIN_SAMPLES_TRAIN,
    DATA_QUALITY_WEIGHTS,
)
from database import get_connection
from utils import find_value_context, clean_string
from log_config import get_logger

logger = get_logger("smart_learner")

# ─── Configurazione ──────────────────────────────────────────────
AUTO_TRAIN_THRESHOLD = 5           # Correzioni per attivare auto-training
AUTO_TRAIN_COOLDOWN_MIN = 10       # Minuti tra auto-training dello stesso campo
PATTERN_MIN_PREFIX_LEN = 8         # Lunghezza minima prefisso pattern
PATTERN_MAX_PREFIX_LEN = 120       # Lunghezza massima prefisso pattern
PATTERN_CONTEXT_WINDOW = 150       # Caratteri contesto per pattern
MAX_LEARNED_PATTERNS = 50          # Massimi pattern per campo
PATTERN_MIN_CONFIDENCE = 0.3       # Confidenza minima per usare un pattern
SELF_EVAL_WINDOW_DAYS = 30         # Finestra per auto-valutazione


# ═════════════════════════════════════════════════════════════════════
# PATTERN LEARNER — Impara DOVE trovare i valori, non QUALI valori
# ═════════════════════════════════════════════════════════════════════

class PatternLearner:
    """Impara pattern strutturali dalle correzioni umane.

    Quando un utente corregge "oggetto_appalto", il sistema:
    1. Trova dove il valore corretto appare nel testo
    2. Estrae il contesto strutturale (prefisso/suffisso)
    3. Generalizza in un pattern riusabile
    4. Su PDF futuri, prova i pattern appresi per estrarre

    Questo funziona anche per valori MAI visti prima perché
    impara la STRUTTURA, non il CONTENUTO.
    """

    def __init__(self):
        self._ensure_tables()

    def _ensure_tables(self):
        """Crea tabelle per pattern appresi."""
        with get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learned_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    field TEXT NOT NULL,
                    prefix_text TEXT NOT NULL,
                    suffix_text TEXT,
                    prefix_regex TEXT,
                    suffix_regex TEXT,
                    success_count INTEGER DEFAULT 1,
                    fail_count INTEGER DEFAULT 0,
                    source TEXT DEFAULT 'correction',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used_at TIMESTAMP,
                    is_active INTEGER DEFAULT 1,
                    UNIQUE(field, prefix_text)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS extraction_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT,
                    field TEXT NOT NULL,
                    extracted_value TEXT,
                    method TEXT,
                    was_corrected INTEGER DEFAULT 0,
                    corrected_value TEXT,
                    pattern_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_lp_field
                ON learned_patterns(field, is_active)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_el_field
                ON extraction_log(field, created_at)
            """)

    # ── Apprendimento Pattern ──────────────────────────────────────

    def learn_from_correction(self, field: str, correct_value: str,
                              wrong_value: str, full_text: str,
                              doc_id: str = "") -> List[dict]:
        """Impara nuovi pattern da una correzione umana.

        Trova dove il valore corretto appare nel testo,
        estrae il contesto strutturale e lo generalizza.
        Ritorna i pattern appresi.
        """
        if not correct_value or not full_text:
            return []

        learned = []
        val_str = str(correct_value).strip()

        # Trova tutte le occorrenze del valore nel testo
        positions = self._find_all_positions(full_text, val_str)

        if not positions:
            # Prova con matching parziale (prime parole significative)
            positions = self._find_partial_positions(full_text, val_str)

        for start, end in positions[:3]:  # Max 3 pattern per correzione
            # Estrai contesto strutturale
            prefix = full_text[max(0, start - PATTERN_CONTEXT_WINDOW):start]
            suffix = full_text[end:min(len(full_text), end + PATTERN_CONTEXT_WINDOW)]

            # Pulisci: prendi l'ultimo "blocco" significativo del prefisso
            prefix = self._extract_meaningful_prefix(prefix)
            suffix = self._extract_meaningful_suffix(suffix)

            if len(prefix) < PATTERN_MIN_PREFIX_LEN:
                continue

            # Genera regex generalizzata
            prefix_regex = self._generalize_to_regex(prefix)
            suffix_regex = self._generalize_to_regex(suffix) if suffix else None

            # Salva pattern
            pattern = self._save_pattern(
                field, prefix, suffix, prefix_regex, suffix_regex
            )
            if pattern:
                learned.append(pattern)
                logger.info(
                    f"Pattern appreso per '{field}': "
                    f"prefix='{prefix[:50]}...' → regex='{prefix_regex[:60]}...'"
                )

        # Logga la correzione
        self._log_extraction(doc_id, field, wrong_value, "corrected",
                             was_corrected=True, corrected_value=correct_value)

        return learned

    def extract_with_patterns(self, field: str, text: str) -> List[Tuple[str, float, int]]:
        """Estrae valori usando pattern appresi.

        Ritorna [(valore, confidenza, pattern_id), ...] ordinati per confidenza.
        Questo funziona per QUALSIASI valore, anche mai visto prima.
        """
        patterns = self._get_active_patterns(field)
        if not patterns:
            return []

        results = []
        for pat in patterns:
            try:
                value = self._apply_pattern(pat, text)
                if value and len(value.strip()) > 2:
                    confidence = self._pattern_confidence(pat)
                    if confidence >= PATTERN_MIN_CONFIDENCE:
                        results.append((
                            clean_string(value),
                            confidence,
                            pat["id"]
                        ))
            except Exception as e:
                logger.debug(f"Pattern {pat['id']} failed for {field}: {e}")

        # Ordina per confidenza decrescente
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def report_success(self, pattern_id: int):
        """Segnala che un pattern ha prodotto un risultato corretto."""
        with get_connection() as conn:
            conn.execute(
                "UPDATE learned_patterns SET success_count = success_count + 1, "
                "last_used_at = CURRENT_TIMESTAMP WHERE id = ?",
                (pattern_id,)
            )

    def report_failure(self, pattern_id: int):
        """Segnala che un pattern ha prodotto un risultato errato."""
        with get_connection() as conn:
            conn.execute(
                "UPDATE learned_patterns SET fail_count = fail_count + 1 WHERE id = ?",
                (pattern_id,)
            )
            # Disattiva pattern con troppi fallimenti
            conn.execute(
                "UPDATE learned_patterns SET is_active = 0 "
                "WHERE id = ? AND fail_count > success_count * 2 AND fail_count >= 3",
                (pattern_id,)
            )

    # ── Pattern Matching ───────────────────────────────────────────

    def _apply_pattern(self, pattern: dict, text: str) -> Optional[str]:
        """Applica un pattern appreso al testo per estrarre un valore."""
        prefix_regex = pattern.get("prefix_regex", "")
        suffix_regex = pattern.get("suffix_regex", "")

        if not prefix_regex:
            return None

        # Costruisci regex: prefisso + cattura + (suffisso opzionale)
        if suffix_regex:
            full_regex = prefix_regex + r"(.+?)" + suffix_regex
        else:
            # Senza suffisso: cattura fino a fine riga o doppio newline
            full_regex = prefix_regex + r"(.+?)(?:\n\n|\n(?=[A-Z]{2,})|\.\s*\n|$)"

        try:
            m = re.search(full_regex, text, re.IGNORECASE | re.DOTALL)
            if m:
                value = m.group(1).strip()
                # Limita lunghezza e pulisci
                if len(value) > 1000:
                    value = value[:1000]
                # Rimuovi trailing punteggiatura inutile
                value = re.sub(r'[\s;,:.]+$', '', value)
                if len(value) > 2:
                    return value
        except re.error:
            pass

        return None

    def _pattern_confidence(self, pattern: dict) -> float:
        """Calcola confidenza di un pattern basata su storico successi/fallimenti."""
        success = pattern.get("success_count", 1)
        fail = pattern.get("fail_count", 0)
        total = success + fail

        if total == 0:
            return 0.5  # Nuovo pattern, confidenza neutra

        # Bayesian: successi / totale con prior
        # Prior: 1 successo + 1 fallimento immaginari (smoothing)
        confidence = (success + 1) / (total + 2)

        # Bonus per pattern molto testati con buon tasso di successo
        if total >= 5 and confidence > 0.7:
            confidence = min(0.95, confidence + 0.05)

        return round(confidence, 3)

    # ── Generalizzazione Pattern ───────────────────────────────────

    def _generalize_to_regex(self, text: str) -> str:
        """Generalizza un frammento di testo in un pattern regex.

        Trasforma dettagli specifici in pattern generici:
        - Numeri → \\d+
        - Date → pattern data flessibile
        - Spazi multipli → \\s+
        - Mantiene parole chiave strutturali
        """
        if not text:
            return ""

        # Escape caratteri speciali regex
        result = re.escape(text.strip())

        # Ri-generalizza pattern comuni (dopo escape)
        # Numeri: \d+ (re.escape li trasforma in letterali)
        result = re.sub(r'(?:\\\d)+', r'\\d+', result)

        # Spazi multipli → \s+
        result = re.sub(r'(?:\\ )+', r'\\s+', result)

        # Newline → \s+
        result = re.sub(r'\\n', r'\\s+', result)

        # Tab → \s+
        result = re.sub(r'\\t', r'\\s+', result)

        # Punti consecutivi (separatori) → \.+\s*
        result = re.sub(r'(?:\\\.\s*){2,}', r'\\.+\\s*', result)

        # Colon/semicolon flessibili: dopo keyword
        result = re.sub(r'\\:', r'[:\\s]', result)

        # Trattini flessibili
        result = re.sub(r'\\-', r'[-–—]', result)

        return result

    def _extract_meaningful_prefix(self, prefix: str) -> str:
        """Estrae la parte significativa del prefisso.

        Cerca l'ultimo marcatore strutturale (keyword, punteggiatura forte)
        e usa quello come inizio del pattern.
        """
        if not prefix:
            return ""

        # Cerca l'ultima keyword strutturale nel prefisso
        structural_markers = [
            r'(?:OGGETTO|APPALTO|PROCEDURA|SERVIZIO|LAVORI|AFFIDAMENTO|FORNITURA)',
            r'(?:Art\.?\s*\d+)',
            r'(?:\d+[.)]\s)',
            r'(?:[A-Z]{2,}\s*:)',  # LABEL: ...
            r'\n\s*\n',  # Doppio newline (inizio paragrafo)
        ]

        best_pos = 0
        for marker in structural_markers:
            for m in re.finditer(marker, prefix, re.IGNORECASE):
                if m.start() > best_pos:
                    best_pos = m.start()

        result = prefix[best_pos:].strip()

        # Troncamento: max lunghezza
        if len(result) > PATTERN_MAX_PREFIX_LEN:
            result = result[-PATTERN_MAX_PREFIX_LEN:]

        return result

    def _extract_meaningful_suffix(self, suffix: str) -> str:
        """Estrae la parte significativa del suffisso."""
        if not suffix:
            return ""

        # Cerca il primo marcatore di fine (keyword, doppio newline)
        end_markers = [
            r'\n\s*\n',
            r'\n(?:CIG|CUP|Art\.?\s*\d+|CAPO|TITOLO|\d+[.)]\s+[A-Z])',
            r'(?:Pag\.?\s*\d+)',
        ]

        end_pos = len(suffix)
        for marker in end_markers:
            m = re.search(marker, suffix, re.IGNORECASE)
            if m and m.start() < end_pos:
                end_pos = m.start()

        result = suffix[:end_pos].strip()

        if len(result) > PATTERN_MAX_PREFIX_LEN:
            result = result[:PATTERN_MAX_PREFIX_LEN]

        return result

    # ── Ricerca Posizioni ──────────────────────────────────────────

    def _find_all_positions(self, text: str, value: str) -> List[Tuple[int, int]]:
        """Trova tutte le posizioni esatte del valore nel testo."""
        positions = []
        start = 0
        val_lower = value.lower()
        text_lower = text.lower()

        while True:
            idx = text_lower.find(val_lower, start)
            if idx < 0:
                break
            positions.append((idx, idx + len(value)))
            start = idx + 1

        return positions

    def _find_partial_positions(self, text: str, value: str) -> List[Tuple[int, int]]:
        """Trova posizioni tramite match parziale (parole chiave)."""
        # Prendi le parole più lunghe e significative
        words = [w for w in value.split() if len(w) > 4]
        if not words:
            words = [w for w in value.split() if len(w) > 2]
        if not words:
            return []

        text_lower = text.lower()

        # Cerca la sequenza di parole più lunga che appare consecutivamente
        best_window = None
        best_word_count = 0

        for i in range(len(words)):
            # Prova a trovare sequenze di parole crescenti
            search_phrase = words[i].lower()
            idx = text_lower.find(search_phrase)
            if idx < 0:
                continue

            # Espandi trovando parole adiacenti
            word_count = 1
            current_end = idx + len(words[i])

            for j in range(i + 1, len(words)):
                next_word = words[j].lower()
                # Cerca nelle vicinanze (entro 50 chars)
                nearby = text_lower[current_end:current_end + 50]
                wpos = nearby.find(next_word)
                if wpos >= 0:
                    word_count += 1
                    current_end = current_end + wpos + len(words[j])
                else:
                    break

            if word_count > best_word_count:
                best_word_count = word_count
                best_window = (idx, current_end)

        if best_window and best_word_count >= 2:
            return [best_window]

        return []

    # ── Persistenza ────────────────────────────────────────────────

    def _save_pattern(self, field: str, prefix: str, suffix: str,
                      prefix_regex: str, suffix_regex: str) -> Optional[dict]:
        """Salva un pattern nel database."""
        with get_connection() as conn:
            c = conn.cursor()

            # Controlla duplicato (stesso prefisso per lo stesso campo)
            existing = c.execute(
                "SELECT id, success_count FROM learned_patterns "
                "WHERE field = ? AND prefix_text = ? AND is_active = 1",
                (field, prefix[:500])
            ).fetchone()

            if existing:
                # Incrementa successo del pattern esistente
                c.execute(
                    "UPDATE learned_patterns SET success_count = success_count + 1, "
                    "last_used_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (existing[0],)
                )
                return {
                    "id": existing[0], "field": field,
                    "status": "reinforced",
                    "success_count": existing[1] + 1
                }

            # Nuovo pattern
            c.execute(
                "INSERT INTO learned_patterns "
                "(field, prefix_text, suffix_text, prefix_regex, suffix_regex) "
                "VALUES (?, ?, ?, ?, ?)",
                (field, prefix[:500], (suffix or "")[:500],
                 prefix_regex[:1000], (suffix_regex or "")[:1000])
            )

            return {
                "id": c.lastrowid, "field": field,
                "status": "new", "prefix": prefix[:80],
                "regex": prefix_regex[:80]
            }

    def _get_active_patterns(self, field: str) -> List[dict]:
        """Recupera pattern attivi per un campo, ordinati per efficacia."""
        with get_connection(readonly=True) as conn:
            rows = conn.execute(
                "SELECT id, prefix_text, suffix_text, prefix_regex, suffix_regex, "
                "success_count, fail_count "
                "FROM learned_patterns "
                "WHERE field = ? AND is_active = 1 "
                "ORDER BY (success_count * 1.0 / (success_count + fail_count + 1)) DESC, "
                "success_count DESC "
                "LIMIT ?",
                (field, MAX_LEARNED_PATTERNS)
            ).fetchall()

        return [{
            "id": r[0], "prefix_text": r[1], "suffix_text": r[2],
            "prefix_regex": r[3], "suffix_regex": r[4],
            "success_count": r[5], "fail_count": r[6],
        } for r in rows]

    def _log_extraction(self, doc_id: str, field: str, value: str,
                        method: str, was_corrected: bool = False,
                        corrected_value: str = None, pattern_id: int = None):
        """Logga un'estrazione per tracking storico."""
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO extraction_log "
                "(doc_id, field, extracted_value, method, was_corrected, "
                "corrected_value, pattern_id) VALUES (?,?,?,?,?,?,?)",
                (doc_id or "", field, (value or "")[:1000], method,
                 1 if was_corrected else 0,
                 (corrected_value or "")[:1000],
                 pattern_id)
            )

    def get_field_stats(self, field: str = None) -> dict:
        """Statistiche pattern appresi per campo."""
        with get_connection(readonly=True) as conn:
            c = conn.cursor()

            if field:
                patterns = c.execute(
                    "SELECT COUNT(*), SUM(success_count), SUM(fail_count) "
                    "FROM learned_patterns WHERE field = ? AND is_active = 1",
                    (field,)
                ).fetchone()
                corrections = c.execute(
                    "SELECT COUNT(*) FROM extraction_log "
                    "WHERE field = ? AND was_corrected = 1",
                    (field,)
                ).fetchone()
                return {
                    "field": field,
                    "active_patterns": patterns[0] or 0,
                    "total_successes": patterns[1] or 0,
                    "total_failures": patterns[2] or 0,
                    "corrections_received": corrections[0] or 0,
                }

            fields = c.execute(
                "SELECT field, COUNT(*), SUM(success_count), SUM(fail_count) "
                "FROM learned_patterns WHERE is_active = 1 GROUP BY field"
            ).fetchall()
            return {
                r[0]: {
                    "active_patterns": r[1],
                    "total_successes": r[2] or 0,
                    "total_failures": r[3] or 0,
                }
                for r in fields
            }


# ═════════════════════════════════════════════════════════════════════
# AUTO TRAINER — Ri-addestramento autonomo progressivo
# ═════════════════════════════════════════════════════════════════════

class AutoTrainer:
    """Gestisce il ri-addestramento automatico dei modelli ML.

    Dopo N correzioni per un campo, attiva automaticamente il training.
    Rispetta cooldown per evitare training eccessivo.
    """

    def __init__(self):
        self._ensure_tables()
        self._last_train_times: Dict[str, datetime] = {}

    def _ensure_tables(self):
        """Crea tabella per tracking auto-training."""
        with get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS auto_train_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    field TEXT NOT NULL,
                    trigger_reason TEXT,
                    corrections_count INTEGER,
                    result_status TEXT,
                    accuracy_before REAL,
                    accuracy_after REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS correction_counter (
                    field TEXT PRIMARY KEY,
                    corrections_since_train INTEGER DEFAULT 0,
                    total_corrections INTEGER DEFAULT 0,
                    last_correction_at TIMESTAMP,
                    last_train_at TIMESTAMP
                )
            """)

    def record_correction(self, field: str) -> Optional[str]:
        """Registra una correzione e verifica se attivare auto-training.

        Ritorna il nome del campo se è il momento di auto-addestrare, None altrimenti.
        """
        with get_connection() as conn:
            c = conn.cursor()

            existing = c.execute(
                "SELECT corrections_since_train FROM correction_counter WHERE field = ?",
                (field,)
            ).fetchone()

            if existing:
                c.execute(
                    "UPDATE correction_counter SET "
                    "corrections_since_train = corrections_since_train + 1, "
                    "total_corrections = total_corrections + 1, "
                    "last_correction_at = CURRENT_TIMESTAMP "
                    "WHERE field = ?",
                    (field,)
                )
                new_count = existing[0] + 1
            else:
                c.execute(
                    "INSERT INTO correction_counter "
                    "(field, corrections_since_train, total_corrections, last_correction_at) "
                    "VALUES (?, 1, 1, CURRENT_TIMESTAMP)",
                    (field,)
                )
                new_count = 1

        # Verifica se è il momento di addestrare
        if new_count >= AUTO_TRAIN_THRESHOLD:
            if self._can_train(field):
                return field

        return None

    def _can_train(self, field: str) -> bool:
        """Verifica cooldown per evitare training troppo frequente."""
        last = self._last_train_times.get(field)
        if last:
            elapsed = datetime.now() - last
            if elapsed < timedelta(minutes=AUTO_TRAIN_COOLDOWN_MIN):
                return False
        return True

    def execute_auto_train(self, field: str) -> dict:
        """Esegue auto-training per un campo.

        Import lazy di ml_engine per evitare import circolare.
        """
        from ml_engine import ml_engine

        logger.info(f"Auto-training attivato per '{field}'")

        # Metriche prima
        old_model = ml_engine.models.get(field)
        accuracy_before = None
        if old_model and old_model.metrics:
            accuracy_before = old_model.metrics.get("accuracy")

        # Training
        result = ml_engine.train_field(field)

        # Metriche dopo
        accuracy_after = None
        if result.get("status") == "ok":
            accuracy_after = result.get("metrics", {}).get("accuracy")

        # Registra
        self._last_train_times[field] = datetime.now()

        with get_connection() as conn:
            conn.execute(
                "INSERT INTO auto_train_log "
                "(field, trigger_reason, corrections_count, result_status, "
                "accuracy_before, accuracy_after) VALUES (?,?,?,?,?,?)",
                (field, "auto_threshold",
                 AUTO_TRAIN_THRESHOLD, result.get("status"),
                 accuracy_before, accuracy_after)
            )
            # Reset contatore correzioni
            conn.execute(
                "UPDATE correction_counter SET "
                "corrections_since_train = 0, last_train_at = CURRENT_TIMESTAMP "
                "WHERE field = ?",
                (field,)
            )

        logger.info(
            f"Auto-training '{field}': {result.get('status')} "
            f"(prima: {accuracy_before}, dopo: {accuracy_after})"
        )

        return result

    def get_status(self) -> dict:
        """Stato dell'auto-trainer con suggerimenti."""
        with get_connection(readonly=True) as conn:
            rows = conn.execute(
                "SELECT field, corrections_since_train, total_corrections, "
                "last_correction_at, last_train_at FROM correction_counter "
                "ORDER BY corrections_since_train DESC"
            ).fetchall()

            recent_trains = conn.execute(
                "SELECT field, result_status, accuracy_before, accuracy_after, "
                "created_at FROM auto_train_log ORDER BY created_at DESC LIMIT 20"
            ).fetchall()

        fields_info = {}
        for r in rows:
            fields_info[r[0]] = {
                "corrections_pending": r[1],
                "total_corrections": r[2],
                "last_correction": r[3],
                "last_train": r[4],
                "progress_to_auto_train": f"{r[1]}/{AUTO_TRAIN_THRESHOLD}",
                "ready_for_auto_train": r[1] >= AUTO_TRAIN_THRESHOLD,
            }

        history = [{
            "field": r[0], "status": r[1],
            "accuracy_before": r[2], "accuracy_after": r[3],
            "timestamp": r[4],
        } for r in recent_trains]

        return {
            "auto_train_threshold": AUTO_TRAIN_THRESHOLD,
            "cooldown_minutes": AUTO_TRAIN_COOLDOWN_MIN,
            "fields": fields_info,
            "recent_trains": history,
        }


# ═════════════════════════════════════════════════════════════════════
# SELF EVALUATOR — Monitora la qualità delle estrazioni nel tempo
# ═════════════════════════════════════════════════════════════════════

class SelfEvaluator:
    """Monitora la qualità delle estrazioni e rileva degradazione.

    Traccia:
    - Tasso di correzione per campo (più alto = peggio)
    - Trend nell'accuratezza dei modelli
    - Campi problematici che necessitano attenzione
    """

    def __init__(self):
        pass  # Usa le stesse tabelle di PatternLearner

    def evaluate_field_quality(self, field: str = None) -> dict:
        """Valuta la qualità di estrazione per campo."""
        with get_connection(readonly=True) as conn:
            c = conn.cursor()

            if field:
                where = "WHERE field = ?"
                where_recent = (
                    f"WHERE field = ? "
                    f"AND created_at > datetime('now', '-{SELF_EVAL_WINDOW_DAYS} days')"
                )
                params = (field,)
            else:
                where = ""
                where_recent = (
                    f"WHERE created_at > datetime('now', '-{SELF_EVAL_WINDOW_DAYS} days')"
                )
                params = ()

            # Totale estrazioni e correzioni
            total = c.execute(
                f"SELECT field, COUNT(*), "
                f"SUM(CASE WHEN was_corrected = 1 THEN 1 ELSE 0 END) "
                f"FROM extraction_log {where} GROUP BY field",
                params
            ).fetchall()

            # Estrazioni recenti (ultimi 30 giorni)
            recent = c.execute(
                f"SELECT field, COUNT(*), "
                f"SUM(CASE WHEN was_corrected = 1 THEN 1 ELSE 0 END) "
                f"FROM extraction_log {where_recent} "
                f"GROUP BY field",
                params
            ).fetchall()

        result = {}
        for r in total:
            fname = r[0]
            total_ext = r[1]
            total_corr = r[2] or 0
            accuracy = 1.0 - (total_corr / total_ext) if total_ext > 0 else None

            result[fname] = {
                "total_extractions": total_ext,
                "total_corrections": total_corr,
                "all_time_accuracy": round(accuracy, 3) if accuracy is not None else None,
            }

        for r in recent:
            fname = r[0]
            if fname in result:
                recent_ext = r[1]
                recent_corr = r[2] or 0
                recent_acc = 1.0 - (recent_corr / recent_ext) if recent_ext > 0 else None
                result[fname]["recent_extractions"] = recent_ext
                result[fname]["recent_corrections"] = recent_corr
                result[fname]["recent_accuracy"] = round(recent_acc, 3) if recent_acc is not None else None

                # Trend: recente vs totale
                all_acc = result[fname]["all_time_accuracy"]
                if all_acc is not None and recent_acc is not None:
                    if recent_acc > all_acc + 0.05:
                        result[fname]["trend"] = "improving"
                    elif recent_acc < all_acc - 0.05:
                        result[fname]["trend"] = "degrading"
                    else:
                        result[fname]["trend"] = "stable"

        return result

    def get_problematic_fields(self) -> List[dict]:
        """Identifica campi con errori frequenti che necessitano attenzione."""
        quality = self.evaluate_field_quality()

        problematic = []
        for field, info in quality.items():
            recent_acc = info.get("recent_accuracy")
            if recent_acc is not None and recent_acc < 0.7:
                problematic.append({
                    "field": field,
                    "recent_accuracy": recent_acc,
                    "recent_corrections": info.get("recent_corrections", 0),
                    "trend": info.get("trend", "unknown"),
                    "recommendation": self._recommend(field, info),
                })

        problematic.sort(key=lambda x: x["recent_accuracy"])
        return problematic

    def _recommend(self, field: str, info: dict) -> str:
        """Genera raccomandazione per un campo problematico."""
        trend = info.get("trend", "unknown")
        total_corr = info.get("total_corrections", 0)

        if trend == "degrading":
            return (
                f"Il campo '{field}' sta peggiorando. "
                f"Verifica le regex di estrazione e i pattern appresi."
            )
        elif total_corr < 5:
            return (
                f"Il campo '{field}' ha poche correzioni ({total_corr}). "
                f"Più correzioni miglioreranno l'apprendimento."
            )
        else:
            return (
                f"Il campo '{field}' ha {total_corr} correzioni. "
                f"Valuta di verificare i pattern nei PDF recenti."
            )


# ═════════════════════════════════════════════════════════════════════
# SMART LEARNER — Orchestratore principale
# ═════════════════════════════════════════════════════════════════════

class SmartLearner:
    """Orchestratore dell'apprendimento progressivo autonomo.

    Coordina:
    - PatternLearner: pattern strutturali per campi con valori unici
    - AutoTrainer: ri-addestramento automatico dopo N correzioni
    - SelfEvaluator: monitoraggio qualità e degradazione

    Usage nel pipeline:
        smart = SmartLearner()

        # Dopo estrazione, prima di restituire risultato
        result = smart.enhance_extraction(result, text)

        # Dopo una correzione umana
        smart.on_correction(field, correct, wrong, text, doc_id)
    """

    def __init__(self):
        self.patterns = PatternLearner()
        self.auto_trainer = AutoTrainer()
        self.evaluator = SelfEvaluator()

    def enhance_extraction(self, result: dict, text: str) -> Tuple[dict, dict]:
        """Migliora risultati usando pattern appresi.

        Per ogni campo vuoto o con confidenza bassa,
        prova i pattern strutturali appresi.

        Ritorna (result_migliorato, metodi_usati).
        """
        pattern_methods = {}

        for field in list(result.keys()):
            if field.startswith('_'):
                continue

            current = result.get(field)
            is_empty = current in [None, "", 0, [], {}]
            # Solo oggetti dict "vuoti" standard (tutte le keys False/None)
            if isinstance(current, dict) and all(
                v in [None, False, ""] for v in current.values()
            ):
                is_empty = True

            if not is_empty:
                continue

            # Prova estrazione con pattern appresi
            candidates = self.patterns.extract_with_patterns(field, text)
            if candidates:
                best_value, best_conf, best_pid = candidates[0]
                result[field] = best_value
                pattern_methods[field] = f"learned_pattern({best_conf:.0%})"

                # Logga successo (provvisorio, confermato se non corretto)
                self.patterns._log_extraction(
                    "", field, best_value, "learned_pattern",
                    pattern_id=best_pid
                )

        return result, pattern_methods

    def on_correction(self, field: str, correct_value: str,
                      wrong_value: str, full_text: str,
                      doc_id: str = "") -> dict:
        """Callback chiamato quando l'utente corregge un campo.

        Azioni:
        1. Apprende nuovi pattern strutturali
        2. Aggiorna contatore correzioni per auto-training
        3. Se soglia raggiunta, attiva auto-training
        4. Segna fallimento di eventuali pattern che hanno generato il valore errato
        """
        response = {
            "patterns_learned": [],
            "auto_train_triggered": False,
            "auto_train_result": None,
        }

        # 1. Apprendi pattern strutturali
        learned = self.patterns.learn_from_correction(
            field, correct_value, wrong_value, full_text, doc_id
        )
        response["patterns_learned"] = learned

        # 2. Segna fallimento pattern errati
        self._mark_failed_patterns(field, wrong_value)

        # 3. Auto-training check
        field_to_train = self.auto_trainer.record_correction(field)
        if field_to_train:
            response["auto_train_triggered"] = True
            try:
                train_result = self.auto_trainer.execute_auto_train(field_to_train)
                response["auto_train_result"] = train_result
            except Exception as e:
                logger.warning(f"Auto-training fallito per '{field}': {e}")
                response["auto_train_result"] = {"status": "error", "message": str(e)}

        return response

    def _mark_failed_patterns(self, field: str, wrong_value: str):
        """Segna come falliti i pattern che hanno generato un valore errato."""
        if not wrong_value:
            return

        with get_connection(readonly=True) as conn:
            # Trova pattern che potrebbero aver generato il valore errato
            recent = conn.execute(
                "SELECT pattern_id FROM extraction_log "
                "WHERE field = ? AND extracted_value = ? AND pattern_id IS NOT NULL "
                "ORDER BY created_at DESC LIMIT 5",
                (field, wrong_value)
            ).fetchall()

        for r in recent:
            if r[0]:
                self.patterns.report_failure(r[0])

    def get_full_status(self) -> dict:
        """Stato completo del sistema di apprendimento progressivo."""
        return {
            "pattern_learner": self.patterns.get_field_stats(),
            "auto_trainer": self.auto_trainer.get_status(),
            "evaluation": self.evaluator.evaluate_field_quality(),
            "problematic_fields": self.evaluator.get_problematic_fields(),
        }


# ═════════════════════════════════════════════════════════════════════
# SINGLETON
# ═════════════════════════════════════════════════════════════════════

smart_learner = SmartLearner()
