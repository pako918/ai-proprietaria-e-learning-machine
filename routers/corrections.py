"""Endpoint per correzioni e training samples CRUD."""

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["Corrections"])


def _get_pipeline():
    from pipeline import pipeline
    return pipeline


@router.post("/api/feedback")
async def record_feedback(payload: dict):
    """Registra correzione utente → training sample (NO auto-training)."""
    doc_id = payload.get("doc_id")
    field = payload.get("field")
    original = payload.get("original", "")
    corrected = payload.get("corrected", "")
    snippet = payload.get("text_snippet", "")
    if not all([doc_id, field, corrected]):
        raise HTTPException(400, "Parametri mancanti: doc_id, field, corrected")
    result = _get_pipeline().record_correction(doc_id, field, original, corrected, snippet)
    return result


@router.get("/api/corrections")
async def get_corrections():
    return _get_pipeline().get_corrections()


@router.get("/api/corrections/stats")
async def get_corrections_stats():
    return _get_pipeline().get_corrections_stats()


@router.put("/api/corrections/{correction_id}")
async def update_correction(correction_id: int, payload: dict):
    sample_id = payload.get("sample_id")
    result = _get_pipeline().update_correction(
        correction_id=correction_id, sample_id=sample_id, data=payload
    )
    if result.get("status") == "error":
        raise HTTPException(400, result["message"])
    return result


@router.delete("/api/corrections/{correction_id}")
async def delete_correction(correction_id: int):
    return _get_pipeline().delete_correction(correction_id=correction_id)


@router.delete("/api/training-sample/{sample_id}")
async def delete_training_sample(sample_id: int):
    return _get_pipeline().delete_correction(sample_id=sample_id)
