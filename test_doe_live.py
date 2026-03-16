"""Test end-to-end del DOE con LLM locale."""
from doe import DOEOrchestrator

doe = DOEOrchestrator()
print("LLM disponibile:", doe.llm.is_available())

# Simulo un'estrazione con campi vuoti
result = {
    "oggetto_appalto": None,
    "cig": None,
    "importo_base_asta": None,
    "stazione_appaltante": None,
    "tipo_procedura": "Procedura aperta",
}

testo = """
DISCIPLINARE DI GARA

STAZIONE APPALTANTE: Comune di Roma - Dipartimento Lavori Pubblici

OGGETTO: Servizi di progettazione definitiva ed esecutiva per la riqualificazione
del Parco Centrale, inclusa direzione lavori e coordinamento sicurezza.

CIG: B1234567890
CUP: J91B21000000001

Importo a base di gara: Euro 450.000,00 oltre IVA
di cui oneri per la sicurezza non soggetti a ribasso: Euro 5.000,00

Procedura aperta ai sensi dell art. 71 del D.Lgs. 36/2023
Criterio di aggiudicazione: offerta economicamente piu vantaggiosa (OEPV)
"""

print("=== TEST RAFFINAMENTO DOE ===\n")
refined = doe.refine_extraction(result, testo, {})
print()

if "_llm_improvements" in refined:
    print("MIGLIORAMENTI LLM:")
    for field, info in refined["_llm_improvements"].items():
        old_val = info.get("old")
        new_val = info.get("new")
        conf = info.get("confidence")
        reason = info.get("reason", "")
        print(f"  {field}: {old_val!r} -> {new_val!r}")
        print(f"    confidenza: {conf}")
        print(f"    motivo: {reason}")
    print(f"\nTotale campi migliorati: {len(refined['_llm_improvements'])}")
else:
    print("Nessun miglioramento rilevato dal LLM")
    for k, v in refined.items():
        if not k.startswith("_"):
            print(f"  {k}: {v!r}")
