import sys, re
sys.path.insert(0, '.')

with open('debug_text.txt', encoding='utf-8') as f:
    text = f.read()
text_lower = text.lower()

print('=== platform URLs ===')
for m in re.finditer(r'https?://[^\s"\'<>\n]+|www\.[^\s"\'<>\n]+', text, re.IGNORECASE):
    print(repr(m.group()[:150]))

print()
print('=== piattaforma section context ===')
for m in re.finditer(r'.{0,100}piattaforma.{0,200}', text, re.IGNORECASE):
    print(repr(m.group()[:300]))
    print()

print()
print('=== universita / stazione appaltante ===')
for m in re.finditer(r'.{0,30}(?:Universit\w+|stazione appaltante).{0,200}', text, re.IGNORECASE):
    print(repr(m.group()[:250]))
    print()
    break

print()
print('=== figure professionali context ===')
for m in re.finditer(r'.{0,50}(?:accompagn|relazione\s+accompagn|direttore\s+dei\s+lavori|coordinatore).{0,200}', text, re.IGNORECASE):
    print(repr(m.group()[:300]))
    print()

print()
print('=== scadenza / termine presentazione ===')
for m in re.finditer(r'.{0,50}(?:scadenza|termine\s+(?:per\s+la\s+)?presentazione|entro\s+il).{0,200}', text, re.IGNORECASE):
    print(repr(m.group()[:300]))
    print()

print()
print('=== RUP extractor context ===')
for m in re.finditer(r'.{0,30}(?:R\.U\.P\.|Responsabile Unico|RUP\b).{0,200}', text, re.IGNORECASE):
    print(repr(m.group()[:300]))
    print()

print()
print('=== importo massimo pagabile global paragraph ===')
idx = text.find("importo massimo pagabile per il LOTTO 1")
if idx == -1:
    idx = text.lower().find("importo massimo pagabile per il lotto 1")
print(repr(text[max(0,idx-200):idx+600]))
