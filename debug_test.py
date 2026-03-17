"""Quick debug test for extraction issues."""
import sys, re, json
sys.path.insert(0, '.')
from extractors.main import extract_rules_based
from json_builder import build_output
result = extract_rules_based('3_TL_2025_Disciplinare_di_Gara.pdf')
output = build_output(result)
if hasattr(output, 'model_dump'):
    out_dict = output.model_dump()
elif hasattr(output, 'dict'):
    out_dict = output.dict()
else:
    out_dict = output

print("TYPE:", type(output))
print("KEYS:", list(out_dict.keys()) if isinstance(out_dict, dict) else "NOT DICT")

print("=== LOTTI ===")
lotti = out_dict.get('lotti_importi', [])
for l in lotti:
    print("Lotto", l.get('lotto'), ": importo=", l.get('importo_euro'), "qf=", l.get('quota_fissa_65_percento_euro'), "qr=", l.get('quota_ribassabile_35_percento_euro'))

print("\n=== PUNTEGGI ===")
cv = out_dict.get('criteri_valutazione', {})
ot = cv.get('offerta_tecnica', {})
oe = cv.get('offerta_economica', {})
print("offerta_tecnica punteggio:", ot.get('punteggio_massimo'))
print("offerta_tecnica criteri count:", len(ot.get('criteri', [])))
print("offerta_economica punteggio:", oe.get('punteggio_massimo'))

print("\n=== FULL OUTPUT (partial) ===")
keys_to_check = ['oggetto', 'RUP', 'RUP_CUC', 'CUP', 'piattaforma_telematica', 'periodo_riferimento']
for k in keys_to_check:
    print(k, ":", json.dumps(out_dict.get(k), ensure_ascii=False))
