"""Endpoint ML Engine: status, quality, data, training, rollback, learning curve."""

from fastapi import APIRouter, HTTPException, Query

from ml_engine import ml_engine as ml_eng

router = APIRouter(tags=["ML Engine"])


def _get_pipeline():
    from pipeline import pipeline
    return pipeline


@router.get("/api/ml/status")
async def ml_status():
    """Stato del motore ML: modelli attivi, dati, configurazione."""
    return _get_pipeline().get_ml_status()


@router.get("/api/ml/quality")
async def ml_quality():
    """Report qualità dati e modelli con raccomandazioni."""
    return _get_pipeline().get_ml_quality()


@router.post("/api/ml/train")
async def ml_train_all():
    """Addestra tutti i modelli con dati sufficienti."""
    return _get_pipeline().train_all()


@router.post("/api/ml/train/{field}")
async def ml_train_field(field: str):
    """Addestra il modello per un campo specifico."""
    return _get_pipeline().train_field(field)


@router.get("/api/ml/learning-curve/{field}")
async def ml_learning_curve(field: str):
    """Curva di apprendimento: come l'accuracy migliora con più dati."""
    return ml_eng.get_learning_curve(field)


@router.get("/api/ml/data")
async def ml_data_stats():
    """Statistiche dataset di training ML."""
    return ml_eng.data.get_data_quality()


@router.get("/api/ml/data/{field}")
async def ml_data_field(field: str):
    """Statistiche dati per un campo specifico."""
    return ml_eng.data.get_data_quality(field)


@router.post("/api/ml/rollback/{field}")
async def ml_rollback(field: str):
    """Ripristina il modello precedente."""
    result = ml_eng.rollback_field(field)
    if result.get("status") == "error":
        raise HTTPException(400, result["message"])
    return result


@router.get("/api/ml/versions")
async def ml_model_versions(field: str = Query(None)):
    """Storico versioni modelli ML."""
    return ml_eng.get_model_versions(field)
