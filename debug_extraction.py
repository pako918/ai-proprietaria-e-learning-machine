"""Debug extraction from the real PDF."""
import sys, json, pprint
sys.path.insert(0, ".")

from extractors.pdf import extract_text_from_pdf
from extractors.main import extract_from_pdf_bytes
from json_builder import build_output

pdf_path = "3_TL_2025_Disciplinare_di_Gara.pdf"

print("=== Extracting text from PDF ===")
pages = extract_text_from_pdf(pdf_path)
full_text = "\n".join(p["text"] for p in pages)
print(f"Total chars: {len(full_text)}")

# Save text for inspection
with open("debug_text.txt", "w", encoding="utf-8") as f:
    f.write(full_text)
print("Saved to debug_text.txt")

print("\n=== Running extractors ===")
raw = extract_all(pdf_path)

print("\n=== suddivisione_lotti ===")
pprint.pprint(raw.get("suddivisione_lotti"))

print("\n=== importi_complessivi ===")
pprint.pprint(raw.get("importi_complessivi"))

print("\n=== info_generali ===")
pprint.pprint(raw.get("informazioni_generali"))

print("\n=== tipo_procedura ===")
pprint.pprint(raw.get("tipo_procedura"))

print("\n=== piattaforma ===")
pprint.pprint(raw.get("piattaforma_telematica"))

print("\n=== requisiti (capacita) ===")
ct = raw.get("requisiti_partecipazione", {}).get("capacita_tecnico_professionale", {})
pprint.pprint(ct)

print("\n=== figure_professionali ===")
fp = raw.get("requisiti_partecipazione", {}).get("figure_professionali", {})
pprint.pprint(fp)

print("\n=== tempistiche ===")
pprint.pprint(raw.get("tempistiche"))

print("\n=== garanzie ===")
pprint.pprint(raw.get("garanzie"))

print("\n=== build_output ===")
out = build_output(raw)
# Show lotti section
dl = out.get("descrizione_lavori_con_importo_totale", {})
for lot in (dl.get("lotti") or []):
    print(f"  Lotto {lot.get('numero')}: importo={lot.get('importo_base_asta')}, quota_fissa={lot.get('quota_fissa_65_percento_euro')}")
