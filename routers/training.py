"""Endpoint per training supervisionato e model versioning."""

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(tags=["Training & Models"])


def _get_pipeline():
    from pipeline import pipeline
    return pipeline


@router.post("/api/train/{field}")
async def train_model(field: str):
    """Training supervisionato — attivato SOLO dall'admin."""
    return _get_pipeline().train_field(field)


@router.post("/api/models/{field}/train")
async def supervised_training(field: str):
    """Alias: training supervisionato."""
    return _get_pipeline().train_field(field)


@router.get("/api/models/versions")
async def model_versions(field: str = Query(None)):
    return _get_pipeline().get_model_versions(field)


@router.post("/api/models/{field}/rollback")
async def model_rollback(field: str):
    result = _get_pipeline().rollback_model(field)
    if result.get("status") == "error":
        raise HTTPException(400, result["message"])
    return result
