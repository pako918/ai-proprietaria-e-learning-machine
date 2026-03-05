"""Test suite per database.py."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import sqlite3
import pytest
from database import get_connection, init_main_tables, init_ml_tables, init_custom_fields_table


class TestGetConnection:
    def test_yields_connection(self):
        with get_connection() as conn:
            assert conn is not None
            assert isinstance(conn, sqlite3.Connection)

    def test_wal_mode(self):
        with get_connection() as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode == "wal"

    def test_foreign_keys_on(self):
        with get_connection() as conn:
            fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
            assert fk == 1

    def test_readonly_prevents_write(self):
        init_main_tables()
        with pytest.raises(Exception):
            with get_connection(readonly=True) as conn:
                conn.execute("INSERT INTO documents (id, filename) VALUES ('test_ro', 'test.pdf')")

    def test_auto_commit(self):
        init_main_tables()
        with get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO documents (id, filename, upload_date) "
                "VALUES ('test_ac', 'test.pdf', '2025-01-01')"
            )
        with get_connection(readonly=True) as conn:
            row = conn.execute("SELECT id FROM documents WHERE id='test_ac'").fetchone()
            assert row is not None
        # Cleanup
        with get_connection() as conn:
            conn.execute("DELETE FROM documents WHERE id='test_ac'")

    def test_rollback_on_error(self):
        init_main_tables()
        try:
            with get_connection() as conn:
                conn.execute(
                    "INSERT INTO documents (id, filename, upload_date) "
                    "VALUES ('test_rb', 'test.pdf', '2025-01-01')"
                )
                raise ValueError("test rollback")
        except ValueError:
            pass
        with get_connection(readonly=True) as conn:
            row = conn.execute("SELECT id FROM documents WHERE id='test_rb'").fetchone()
            assert row is None


class TestInitTables:
    def test_init_main_tables(self):
        init_main_tables()
        with get_connection(readonly=True) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        assert "documents" in tables
        assert "training_samples" in tables
        assert "model_versions" in tables
        assert "feedback_log" in tables

    def test_init_ml_tables(self):
        init_ml_tables()
        with get_connection(readonly=True) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        assert "ml_training_data" in tables
        assert "ml_model_registry" in tables
        assert "ml_predictions_log" in tables

    def test_init_custom_fields_table(self):
        init_custom_fields_table()
        with get_connection(readonly=True) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        assert "custom_fields" in tables

    def test_idempotent(self):
        """init_ functions should be safe to call multiple times."""
        init_main_tables()
        init_main_tables()
        init_ml_tables()
        init_ml_tables()
        init_custom_fields_table()
        init_custom_fields_table()
