"""Endpoint per documenti: history, stats, document text, PDF info."""

import json
from fastapi import APIRouter, HTTPException

from database import get_connection

router = APIRouter(tags=["Documents"])


def _get_pipeline():
    from pipeline import pipeline
    return pipeline


@router.get("/api/stats")
async def get_stats():
    try:
        return _get_pipeline().get_stats()
    except Exception as e:
        raise HTTPException(500, f"Errore recupero statistiche: {e}")


@router.get("/api/history")
async def get_history():
    try:
        return _get_pipeline().get_history()
    except Exception as e:
        raise HTTPException(500, f"Errore recupero storico: {e}")


@router.get("/api/history/{doc_id}")
async def get_document(doc_id: str):
    try:
        with get_connection(readonly=True) as conn:
            row = conn.execute(
                "SELECT filename, upload_date, extracted_json, corrected_json "
                "FROM documents WHERE id=?",
                (doc_id,)
            ).fetchone()
    except Exception as e:
        raise HTTPException(500, f"Errore database: {e}")
    if not row:
        raise HTTPException(404, "Documento non trovato")
    return {
        "id": doc_id,
        "filename": row[0],
        "upload_date": row[1],
        "extracted": json.loads(row[2]) if row[2] else {},
        "corrected": json.loads(row[3]) if row[3] else {},
    }


@router.get("/api/document/{doc_id}/text")
async def get_doc_text(doc_id: str):
    data = _get_pipeline().get_document_text(doc_id)
    if not data:
        raise HTTPException(404, "Documento non trovato")
    return data


@router.get("/api/pdf-info/{doc_id}")
async def get_pdf_info(doc_id: str):
    """Info strutturali PDF (tabelle, chunks, metadati)."""
    pipe = _get_pipeline()
    # Cerca il documento specifico nella cache per doc_id o filename
    parsed = None
    for fname, p in pipe._last_parsed.items():
        if fname == doc_id or getattr(p, 'filename', '') == doc_id:
            parsed = p
            break
    # Fallback: se c'è un solo documento nella cache, lo usa
    if parsed is None and len(pipe._last_parsed) == 1:
        parsed = next(iter(pipe._last_parsed.values()))
    if parsed is None:
        raise HTTPException(404, "Documento non in cache")
    return {
        "filename": parsed.filename,
        "is_native": parsed.is_native,
        "total_pages": parsed.total_pages,
        "parser_used": parsed.parser_used,
        "metadata": parsed.metadata,
        "warnings": parsed.warnings,
        "tables": parsed.tables_json[:20],
        "chunks": [
            {"type": c.chunk_type, "title": c.title,
             "pages": f"{c.page_start}-{c.page_end}",
             "section": c.section_path,
             "length": len(c.content)}
            for c in parsed.chunks[:50]
        ],
    }
