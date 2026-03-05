"""
AppaltoAI — Gestione Database Centralizzata
Context manager per connessioni SQLite, inizializzazione e migrazioni.
"""

import sqlite3
import logging
from contextlib import contextmanager
from config import DB_PATH

logger = logging.getLogger("appaltoai.database")


@contextmanager
def get_connection(readonly: bool = False):
    """Context manager per connessioni SQLite con commit/rollback automatico.

    Uso:
        with get_connection() as conn:
            conn.execute("INSERT INTO ...", params)
        # commit automatico all'uscita, rollback su eccezione
    """
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    if readonly:
        conn.execute("PRAGMA query_only=ON")
    try:
        yield conn
        if not readonly:
            conn.commit()
    except Exception:
        if not readonly:
            conn.rollback()
        raise
    finally:
        conn.close()


def init_main_tables():
    """Crea le tabelle principali della pipeline."""
    with get_connection() as conn:
        conn.executescript("""
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
        _run_migrations(conn)


def init_ml_tables():
    """Crea le tabelle del motore ML."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS ml_training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                field TEXT NOT NULL,
                text_snippet TEXT NOT NULL,
                correct_value TEXT NOT NULL,
                wrong_value TEXT,
                source TEXT DEFAULT 'auto',
                quality_score REAL DEFAULT 0.5,
                doc_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                is_active INTEGER DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_ml_field ON ml_training_data(field);
            CREATE INDEX IF NOT EXISTS idx_ml_active ON ml_training_data(is_active);

            CREATE TABLE IF NOT EXISTS ml_model_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                field TEXT NOT NULL,
                version INTEGER NOT NULL,
                accuracy REAL,
                f1_score REAL,
                cv_score_mean REAL,
                cv_score_std REAL,
                n_samples INTEGER,
                n_classes INTEGER,
                training_time_ms REAL,
                feature_importance TEXT,
                is_active INTEGER DEFAULT 1,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_model_field ON ml_model_registry(field);

            CREATE TABLE IF NOT EXISTS ml_predictions_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                field TEXT,
                predicted_value TEXT,
                confidence REAL,
                method TEXT,
                was_correct INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)


def init_custom_fields_table():
    """Crea la tabella per i campi custom del field_registry."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS custom_fields (
                key TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                icon TEXT DEFAULT '📋',
                category TEXT DEFAULT 'Custom',
                field_type TEXT DEFAULT 'text',
                mono INTEGER DEFAULT 0,
                highlight INTEGER DEFAULT 0,
                full_width INTEGER DEFAULT 0,
                patterns TEXT DEFAULT '[]',
                validator_type TEXT DEFAULT 'text',
                description TEXT DEFAULT '',
                extraction_hint TEXT DEFAULT '',
                created_at TEXT,
                updated_at TEXT
            )
        """)


def _run_migrations(conn: sqlite3.Connection):
    """Migrazioni sicure per colonne opzionali."""
    migrations = [
        ("documents", "full_text", "TEXT"),
        ("documents", "model_version", "TEXT"),
        ("training_samples", "source", 'TEXT DEFAULT "manual"'),
        ("model_versions", "is_active", "INTEGER DEFAULT 1"),
        ("model_versions", "notes", "TEXT"),
    ]
    for table, column, col_type in migrations:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        except sqlite3.OperationalError:
            pass  # Colonna già esistente
