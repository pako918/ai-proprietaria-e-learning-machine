"""Endpoint core: root, health, validate, pipeline status."""

import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse

from config import BASE_DIR, API_VERSION
from schemas import full_validation

router = APIRouter()


def _get_pipeline():
    from pipeline import pipeline
    return pipeline


@router.get("/")
async def root():
    index_file = BASE_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return {"status": "AppaltoAI attivo", "version": API_VERSION}


@router.get("/api/health")
async def health():
    pipe = _get_pipeline()
    stats = pipe.get_stats()
    return {
        "status": "ok",
        "version": API_VERSION,
        "architecture": "ML-powered extraction pipeline",
        "learning_mode": "data-driven supervised",
        "ml_engine": pipe.get_ml_status(),
        "stats": stats,
    }


@router.get("/api/pipeline/status")
async def pipeline_status():
    pipe = _get_pipeline()
    stats = pipe.get_stats()
    versions = pipe.get_model_versions()
    ml_status = pipe.get_ml_status()
    return {
        "pipeline_version": API_VERSION,
        "architecture": "ML-powered extraction",
        "learning_mode": "progressive autonomous",
        "ml_engine": ml_status,
        "stats": stats,
        "models": versions,
    }


@router.post("/api/validate")
async def validate_result(payload: dict):
    try:
        validation = full_validation(payload)
        return JSONResponse(content=validation)
    except Exception as e:
        raise HTTPException(500, f"Errore validazione: {str(e)}")
