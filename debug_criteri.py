"""Debug: analizza i criteri estratti per capire perché la somma è 83 invece di 80."""
import glob
from extractors import extract_text_from_pdf, extract_rules_based
from json_builder import build_output_with_methods

pdfs = glob.glob("data/uploads/*.pdf") + glob.glob("uploads/*.pdf") + glob.glob("*.pdf")
if not pdfs:
    print("No PDF found")
    exit()

pdf = pdfs[0]
print(f"PDF: {pdf}")
text = extract_text_from_pdf(pdf)
nested = extract_rules_based(text)

ot = nested.get("criteri_valutazione", {}).get("offerta_tecnica", {})
criteri = ot.get("criteri", [])
pt_max = ot.get("punteggio_massimo")

print(f"\npunteggio_massimo: {pt_max}")
print(f"Numero totale criteri (flat): {len(criteri)}")
print()

print("=== TUTTI I CRITERI (flat list da extract_rules_based) ===")
for c in criteri:
    print(f"  {c['codice']:6s}  livello={c['livello']:15s}  punteggio={c['punteggio']:5.1f}  nome={c['nome'][:70]}")

print()
criteri_top = [c for c in criteri if c["livello"] == "criterio"]
criteri_sub = [c for c in criteri if c["livello"] == "sub_criterio"]

somma_top = sum(c["punteggio"] for c in criteri_top)
somma_sub = sum(c["punteggio"] for c in criteri_sub)

print(f"=== CRITERI TOP-LEVEL (livello='criterio') ===")
for c in criteri_top:
    print(f"  {c['codice']:6s}  punteggio={c['punteggio']:5.1f}  nome={c['nome'][:70]}")
print(f"  SOMMA: {somma_top}")

print()
print(f"=== SUB-CRITERI (livello='sub_criterio') ===")
for c in criteri_sub:
    print(f"  {c['codice']:6s}  punteggio={c['punteggio']:5.1f}  nome={c['nome'][:70]}")
print(f"  SOMMA: {somma_sub}")

# Check for duplicates
print()
print("=== DUPLICATI PER CODICE ===")
from collections import Counter
codes = [c["codice"] for c in criteri]
dupes = {code: count for code, count in Counter(codes).items() if count > 1}
if dupes:
    print(f"  DUPLICATI TROVATI: {dupes}")
    for code, count in dupes.items():
        entries = [c for c in criteri if c["codice"] == code]
        for e in entries:
            print(f"    {e['codice']}  punteggio={e['punteggio']}  livello={e['livello']}  nome={e['nome'][:60]}")
else:
    print("  Nessun duplicato")

# Now check json_builder output
print()
print("=== OUTPUT DOPO json_builder ===")
result, methods = build_output_with_methods(nested)
ot_final = result.get("offerta_tecnica", {})
if ot_final:
    criteri_final = ot_final.get("criteri", [])
    pt_max_final = ot_final.get("punteggio_massimo")
    somma_final = sum(c.get("punteggio", 0) or 0 for c in criteri_final)
    print(f"  punteggio_massimo: {pt_max_final}")
    print(f"  Numero criteri: {len(criteri_final)}")
    for c in criteri_final:
        p = c.get("punteggio", 0)
        nome = c.get("nome", c.get("codice", "?"))
        cod = c.get("codice", "?")
        print(f"    {cod:6s}  punteggio={p:5.1f}  nome={nome[:60]}")
    print(f"  SOMMA: {somma_final}")
    print(f"  DIFFERENZA con max: {somma_final - pt_max_final}")
