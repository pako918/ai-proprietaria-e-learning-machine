"""Quick debug test for extraction issues."""
import sys, json
sys.path.insert(0, '.')
from extractors.main import extract_rules_based
from json_builder import build_output
result = extract_rules_based('3_TL_2025_Disciplinare_di_Gara.pdf')
out_dict = build_output(result)

print("TOP-LEVEL KEYS:", list(out_dict.keys()))

dl = out_dict.get('descrizione_lavori_con_importo_totale', {})
print("DescrizioneLavori keys:", list(dl.keys()) if dl else "None")
lotti = (dl or {}).get('lotti', [])
print("LOTTI:")
for l in lotti:
    print("  Lotto", l.get('lotto'), ": importo=", l.get('importo_euro'), "qf=", l.get('quota_fissa_65_percento_euro'), "qr=", l.get('quota_ribassabile_35_percento_euro'))

print()
print("=== CRITERI ===")
cv = out_dict.get('criteri_valutazione_offerta_tecnica')
print("criteri_valutazione_offerta_tecnica:", json.dumps(cv, ensure_ascii=False, default=str)[:500] if cv else "None")

print()
print("=== SA / RUP ===")
sa = out_dict.get('stazione_appaltante', {})
print("SA:", json.dumps(sa, ensure_ascii=False, default=str)[:400] if sa else "None")
