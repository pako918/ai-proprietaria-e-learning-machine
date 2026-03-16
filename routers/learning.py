"""Endpoint per smart learner e endpoint legacy."""

from fastapi import APIRouter

from smart_learner import smart_learner as sl

router = APIRouter(tags=["Learning"])


@router.get("/api/learning/status")
async def learning_status():
    """Stato completo del sistema di apprendimento progressivo."""
    return sl.get_full_status()


@router.get("/api/learning/patterns/{field}")
async def learning_patterns(field: str):
    """Pattern strutturali appresi per un campo specifico."""
    return sl.patterns.get_field_stats(field)


@router.get("/api/learning/evaluation")
async def learning_evaluation():
    """Valutazione qualità estrazioni con trend e campi problematici."""
    return {
        "field_quality": sl.evaluator.evaluate_field_quality(),
        "problematic_fields": sl.evaluator.get_problematic_fields(),
    }


@router.get("/api/learning/auto-train")
async def auto_train_status():
    """Stato dell'auto-trainer: correzioni pendenti, soglie, storico."""
    return sl.auto_trainer.get_status()


# ── LEGACY COMPATIBILITY STUBS ────────────────────────────────────────

@router.get("/api/quarantine")
async def get_quarantine():
    return []


@router.get("/api/auto-learn/stats")
async def auto_learn_stats():
    return sl.auto_trainer.get_status()


@router.post("/api/quarantine/{qid}/approve")
async def approve_quarantine(qid: int):
    return {"status": "deprecated", "message": "Quarantine rimossa. Usa correzioni manuali."}


@router.post("/api/quarantine/{qid}/reject")
async def reject_quarantine(qid: int):
    return {"status": "deprecated", "message": "Quarantine rimossa."}
