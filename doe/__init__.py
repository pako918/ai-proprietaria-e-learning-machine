"""DOE — Directive-Orchestration-Execution Layer con LLM Locale.

Architettura a 3 livelli:
  1. Directive Layer   → SOP in Markdown (doe/directives/)
  2. Orchestration Layer → Agente AI con ciclo ReAct (LLM locale via Ollama)
  3. Execution Layer   → Script Python deterministici (regex, ML, pattern)

Self-learning loop:
  Esecuzione → Errore → Correzione → Evoluzione delle direttive

Utilizzo nella pipeline:
    from doe import DOEOrchestrator, SelfLearner

    doe = DOEOrchestrator()
    result = doe.refine_extraction(result, text, methods)
"""

from .llm_local import OllamaClient
from .directives import DirectiveManager
from .orchestrator import DOEOrchestrator
from .self_learner import SelfLearner
from .tools import execute_tool, get_tools_schema

__all__ = [
    "OllamaClient",
    "DirectiveManager",
    "DOEOrchestrator",
    "SelfLearner",
    "execute_tool",
    "get_tools_schema",
]
