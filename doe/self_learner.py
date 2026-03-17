"""Self-Learning Engine — Ciclo: Esecuzione → Errore → Correzione → Evoluzione.

Integrato con il DOE Orchestrator:
1. ESECUZIONE  — la pipeline processa il documento
2. ERRORE      — l'agente rileva campi vuoti / bassa confidenza / anomalie
3. CORREZIONE  — l'LLM locale propone correzioni ragionate
4. EVOLUZIONE  — le direttive vengono aggiornate per evitare l'errore in futuro

Le direttive apprese finiscono in doe/directives/learned/*.md
e vengono automaticamente incluse nel prompt dell'agente.
"""

import json
import logging
from datetime import datetime

from .config import EVOLUTION_MIN_CORRECTIONS, LEARNED_DIR
from .llm_local import OllamaClient
from .directives import DirectiveManager

log = logging.getLogger(__name__)


class SelfLearner:
    """Motore di auto-apprendimento basato sull'evoluzione delle direttive."""

    def __init__(self, llm: OllamaClient | None = None,
                 directives: DirectiveManager | None = None):
        self.llm = llm or OllamaClient()
        self.directives = directives or DirectiveManager()
        # Buffer correzioni per campo → lista di {wrong, correct, context, ...}
        self._correction_buffer: dict[str, list[dict]] = {}

    # ── 1. DETECT: Identifica problemi ────────────────────────────

    def evaluate_extraction(self, result: dict, text: str,
                            methods: dict | None = None) -> list[dict]:
        """Valuta un'estrazione e identifica campi problematici."""
        problems: list[dict] = []
        methods = methods or {}
        for field, value in _iter_fields(result):
            issue = None
            # Campo vuoto in un documento lungo → probabile errore
            if (value is None or value == "") and len(text) > 5000:
                issue = {"type": "empty", "severity": "medium"}
            # Bassa confidenza
            if field in methods and isinstance(methods[field], dict):
                conf = methods[field].get("confidence", 1.0)
                if conf < 0.4:
                    issue = {"type": "low_confidence", "severity": "high",
                             "confidence": conf}
            if issue:
                issue["field"] = field
                issue["current_value"] = value
                problems.append(issue)
        return problems

    # ── 2. CORRECT: Proponi correzioni via LLM ────────────────────

    def propose_corrections(self, problems: list[dict],
                            text: str) -> list[dict]:
        """Per ogni problema, chiede all'LLM locale di proporre un fix."""
        if not self.llm.is_available() or not problems:
            return []

        proposals: list[dict] = []
        for problem in problems[:5]:  # Max 5 per ciclo
            field = problem["field"]
            section = self._extract_relevant_section(text, field)

            prompt = (
                f'Devo estrarre il campo "{field}" da un disciplinare di gara.\n\n'
                f'Problema: {problem["type"]}\n'
                f'Valore attuale: {json.dumps(problem.get("current_value"), ensure_ascii=False)}\n\n'
                f'Sezione rilevante del documento:\n---\n{section}\n---\n\n'
                f'Estrai il valore corretto. Se non è presente, rispondi null.\n\n'
                f'Rispondi in JSON: '
                f'{{"value": ..., "confidence": 0.0-1.0, "evidence": "testo esatto dal documento"}}'
            )
            resp = self.llm.generate_json(prompt)
            if resp and resp.get("value") is not None:
                proposals.append({
                    "field": field,
                    "problem": problem,
                    "proposed_value": resp["value"],
                    "confidence": resp.get("confidence", 0.5),
                    "evidence": resp.get("evidence", ""),
                })
        return proposals

    # ── 3. RECORD: Registra correzioni per evoluzione futura ──────

    def record_correction(self, field: str, wrong_value, correct_value,
                          text_context: str, reason: str = ""):
        """Registra una correzione; se il buffer è pieno → evolve direttiva."""
        if field not in self._correction_buffer:
            self._correction_buffer[field] = []

        self._correction_buffer[field].append({
            "wrong": wrong_value,
            "correct": correct_value,
            "context": text_context[:500],
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        })

        if len(self._correction_buffer[field]) >= EVOLUTION_MIN_CORRECTIONS:
            self._evolve_directive(field)

    # ── 4. EVOLVE: Aggiorna direttive ────────────────────────────

    def _evolve_directive(self, field: str):
        """Genera una nuova direttiva basata sulle correzioni accumulate.

        PRINCIPIO: la direttiva impara DOVE trovare il valore nel documento
        (sezione, etichetta precedente, struttura, pattern testuale), non
        QUALE valore aspettarsi. I valori specifici sono validi solo per il
        documento da cui provengono e non vanno inclusi come esempi fissi.
        """
        corrections = self._correction_buffer.get(field, [])
        if not corrections or not self.llm.is_available():
            return

        # Esporta solo il CONTESTO strutturale, non i valori specifici.
        # Il sistema impara il pattern "dove cercare", non "cosa cercare".
        context_examples = "\n".join(
            f"- Contesto rilevante nel documento:\n"
            f"  ...{c['context'][:200]}..."
            for c in corrections[-10:]
        )
        current_directive = (
            self.directives.get_field_directive(field) or "Nessuna direttiva esistente"
        )

        prompt = (
            f'Analizza questi contesti documentali per il campo "{field}" '
            f"e genera una direttiva migliorata per l'estrazione.\n\n"
            f"Direttiva attuale:\n{current_directive}\n\n"
            f"Contesti documentali in cui il campo era presente "
            f"(non i valori specifici, che cambiano per ogni documento):\n"
            f"{context_examples}\n\n"
            f"Genera una nuova direttiva in Markdown che:\n"
            f"1. Descriva in quale sezione del documento si trova tipicamente "
            f"il campo\n"
            f"2. Indichi le etichette testuali o i titoli che lo precedono\n"
            f"3. Descriva la struttura (tabella, paragrafo, intestazione) "
            f"che contiene il dato\n"
            f"4. Fornisca pattern testuali o numerici per identificare "
            f"il valore (es. formato, unità di misura)\n"
            f"5. Elenchi errori di localizzazione da evitare\n\n"
            f"IMPORTANTE: non inserire valori specifici come esempi — "
            f"ogni documento ha i propri valori.\n\n"
            f"Rispondi SOLO con la direttiva in Markdown."
        )
        new_directive = self.llm.generate(prompt, temperature=0.2)

        if new_directive and len(new_directive) > 50:
            reason = f"Evoluzione dopo {len(corrections)} correzioni"
            self.directives.save_learned_directive(field, new_directive, reason)
            self._log_evolution(field, corrections, new_directive)
            self._correction_buffer[field] = []
            log.info("Direttiva evoluta per '%s' dopo %d correzioni",
                     field, len(corrections))

    def _log_evolution(self, field: str, corrections: list, new_directive: str):
        """Registra l'evoluzione per audit trail."""
        log_path = LEARNED_DIR / "_evolution_log.jsonl"
        entry = {
            "field": field,
            "timestamp": datetime.now().isoformat(),
            "num_corrections": len(corrections),
            "directive_preview": new_directive[:200],
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── Utility ───────────────────────────────────────────────────

    @staticmethod
    def _extract_relevant_section(text: str, field: str) -> str:
        """Estrae la sezione del testo più rilevante per un campo."""
        field_keywords: dict[str, list[str]] = {
            "oggetto_appalto": ["oggetto", "affidamento", "appalto"],
            "stazione_appaltante": ["stazione appaltante", "ente", "committente"],
            "importo_base_asta": ["importo", "base d'asta", "valore"],
            "cig": ["cig", "codice identificativo"],
            "cup": ["cup", "codice unico"],
            "tipo_procedura": ["procedura", "tipo di gara"],
            "criterio_aggiudicazione": ["criterio", "aggiudicazione", "oepv"],
            "durata_contratto": ["durata", "termine", "contratto"],
            "garanzia_provvisoria": ["garanzia", "provvisoria", "cauzione"],
            "subappalto": ["subappalto", "subappaltare"],
        }
        keywords = field_keywords.get(field, [field.replace("_", " ")])
        text_lower = text.lower()
        for kw in keywords:
            idx = text_lower.find(kw)
            if idx != -1:
                start = max(0, idx - 200)
                return text[start : start + 3000]
        return text[:3000]


def _iter_fields(d: dict, prefix: str = ""):
    """Itera su tutti i campi di un dict annidato."""
    for k, v in d.items():
        if k.startswith("_"):
            continue
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            yield from _iter_fields(v, path)
        else:
            yield path, v
