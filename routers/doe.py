"""Router per il DOE (Directive-Orchestration-Execution) Layer.

Endpoint per status, test LLM locale, e gestione direttive.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/api/doe", tags=["DOE"])


@router.get("/status")
def doe_status():
    """Stato del DOE Layer: LLM, modelli, direttive caricate."""
    from doe import DOEOrchestrator

    doe = DOEOrchestrator()
    available = doe.llm.is_available()
    models = doe.llm.list_models() if available else []

    return {
        "ollama_available": available,
        "ollama_host": doe.llm.host,
        "model_configured": doe.llm.model,
        "models_installed": models,
        "directives_loaded": len(doe.directives._cache),
        "system_prompt_length": len(doe.system_prompt),
        "setup_instructions": None if available else {
            "step_1": "Installa Ollama da https://ollama.com",
            "step_2": "Apri un terminale e avvia: ollama serve",
            "step_3": f"Scarica un modello: ollama pull {doe.llm.model}",
            "step_4": "Riavvia AppaltoAI",
            "modelli_consigliati": [
                "mistral (7B - veloce, buon italiano)",
                "qwen2.5:7b (7B - ottimo multilingue)",
                "llama3:8b (8B - buona qualità generale)",
                "gemma2:9b (9B - buon ragionamento)",
            ],
        },
    }


@router.post("/test")
def doe_test(text: str = "Oggetto dell'appalto: Servizi di manutenzione. CIG: A123456789."):
    """Testa il raffinamento DOE su un testo di esempio."""
    from doe import DOEOrchestrator

    doe = DOEOrchestrator()
    if not doe.llm.is_available():
        return {"error": "Ollama non disponibile. Installa e avvia Ollama."}

    dummy_result = {
        "oggetto_appalto": None,
        "cig": None,
    }
    refined = doe.refine_extraction(dummy_result, text, {})
    return {
        "original": dummy_result,
        "refined": refined,
        "improvements": refined.get("_llm_improvements", {}),
    }


@router.get("/directives")
def list_directives():
    """Elenca le direttive caricate (SOP + learned)."""
    from doe import DirectiveManager

    dm = DirectiveManager()
    return {
        "directives": list(dm._cache.keys()),
        "learned": [k for k in dm._cache if k.startswith("learned/")],
        "total": len(dm._cache),
    }
