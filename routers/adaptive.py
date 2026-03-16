"""Endpoint per l'Adaptive Learner — monitoraggio apprendimento autonomo."""

from fastapi import APIRouter

from adaptive_learner import adaptive_learner as al

router = APIRouter(tags=["Adaptive Learning"])


@router.get("/api/adaptive/status")
async def adaptive_status():
    """Stato completo del sistema di apprendimento adattivo.

    Mostra: documenti processati, estrazioni in memoria,
    regole auto-generate, auto-correzioni, accuratezza per campo.
    """
    return al.get_learning_status()


@router.get("/api/adaptive/field/{field}")
async def adaptive_field_intelligence(field: str):
    """Report dettagliato sull'intelligenza acquisita per un campo.

    Mostra: estrazioni recenti, statistiche, regole auto-generate,
    livello di maturità dell'apprendimento.
    """
    return al.get_field_intelligence(field)


@router.get("/api/adaptive/similar/{doc_id}")
async def find_similar_docs(doc_id: str):
    """Trova i documenti più simili a quello specificato."""
    from database import get_connection
    with get_connection(readonly=True) as conn:
        row = conn.execute(
            "SELECT full_text, extracted_json FROM documents WHERE id = ?",
            (doc_id,)
        ).fetchone()
    if not row:
        return {"error": "Documento non trovato"}

    import json
    text = row[0] or ""
    result = json.loads(row[1]) if row[1] else {}
    similar = al.doc_sim.find_similar(text, result, exclude_doc=doc_id)
    return {"doc_id": doc_id, "similar_documents": similar}


@router.get("/api/adaptive/rules")
async def list_auto_rules():
    """Lista tutte le regole auto-generate attive."""
    from database import get_connection
    with get_connection(readonly=True) as conn:
        rules = conn.execute(
            "SELECT id, field, rule_type, regex_pattern, "
            "success_count, fail_count, source_doc_count, "
            "is_active, created_at, last_used "
            "FROM auto_rules ORDER BY success_count DESC"
        ).fetchall()
    return [{
        "id": r[0], "field": r[1], "type": r[2],
        "regex": (r[3] or "")[:120], "success": r[4],
        "fail": r[5], "from_docs": r[6], "active": bool(r[7]),
        "created": r[8], "last_used": r[9],
    } for r in rules]


@router.post("/api/adaptive/generate-rules")
async def trigger_rule_generation():
    """Forza la generazione di nuove regole per tutti i campi."""
    al._trigger_rule_generation()
    return al.get_learning_status()


@router.get("/api/adaptive/corrections-log")
async def auto_corrections_log(limit: int = 50):
    """Log delle auto-correzioni effettuate dal sistema."""
    from database import get_connection
    with get_connection(readonly=True) as conn:
        rows = conn.execute(
            "SELECT id, doc_id, field, original_value, corrected_value, "
            "reason, confidence, created_at "
            "FROM auto_corrections_log "
            "ORDER BY created_at DESC LIMIT ?",
            (min(limit, 200),)
        ).fetchall()
    return [{
        "id": r[0], "doc_id": r[1], "field": r[2],
        "original": r[3], "corrected": r[4],
        "reason": r[5], "confidence": r[6], "timestamp": r[7],
    } for r in rows]


@router.get("/api/adaptive/field-stats")
async def all_field_stats():
    """Statistiche di tutti i campi monitorati."""
    from database import get_connection
    import json
    with get_connection(readonly=True) as conn:
        rows = conn.execute(
            "SELECT field, value_type, num_extractions, num_unique_values, "
            "num_corrections, avg_numeric, stddev_numeric, "
            "min_numeric, max_numeric, avg_text_length "
            "FROM field_value_stats ORDER BY num_extractions DESC"
        ).fetchall()
    return [{
        "field": r[0], "type": r[1], "extractions": r[2],
        "unique_values": r[3], "corrections": r[4],
        "accuracy": round((1 - r[4] / r[2]) * 100, 1) if r[2] > 0 else None,
        "numeric_stats": {
            "avg": r[5], "stddev": r[6],
            "min": r[7], "max": r[8],
        } if r[5] is not None else None,
        "avg_text_length": r[9],
    } for r in rows]
