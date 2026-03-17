import sys, pprint
sys.path.insert(0, '.')
from extractors.main import extract_rules_based

with open('debug_text.txt', encoding='utf-8') as f:
    text = f.read()

raw = extract_rules_based(text)

print('=== suddivisione_lotti (importi) ===')
for l in raw.get('suddivisione_lotti', {}).get('lotti', []):
    n = l['numero']
    imp = l.get('importo_base_asta')
    print(f'  Lotto {n}: importo={imp}')

print()
print('=== piattaforma ===')
pprint.pprint(raw.get('piattaforma_telematica'))

print()
print('=== stazione_appaltante ===')
pprint.pprint(raw.get('informazioni_generali', {}).get('stazione_appaltante'))

print()
print('=== figure_professionali ===')
pprint.pprint(raw.get('requisiti_partecipazione', {}).get('gruppo_di_lavoro', {}).get('figure_professionali'))
