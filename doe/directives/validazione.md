# Regole di Validazione

## Importi
- `importo_base_asta` deve essere > 0
- Se ci sono lotti, la somma degli importi dei lotti ≈ importo totale (tolleranza 5%)
- `oneri_sicurezza` ≤ 10% dell'importo base (di norma)
- Importi in formato numerico puro (no €, no punti delle migliaia)

## Codici
- CIG: esattamente 10 caratteri alfanumerici
- CUP: 15 caratteri, inizia con una lettera maiuscola
- CPV: formato "XXXXXXXX-X" (8 cifre + trattino + 1 cifra)

## Date
- Formato preferito: YYYY-MM-DD
- Le scadenze devono essere nel futuro rispetto alla data del bando
- La scadenza presentazione offerte > data pubblicazione

## Coerenza Interna
- Se `criterio_aggiudicazione` = "OEPV":
  - Devono esistere criteri tecnici con punteggi
  - La somma dei punteggi tecnici + economici = 100 (di norma)
- Se ci sono lotti:
  - Ogni lotto deve avere almeno un CIG
  - Ogni lotto deve avere un importo

## Anomalie
- Importi negativi → errore
- Durate > 120 mesi → verificare (probabilmente sbagliato)
- Percentuali > 100% → errore
- Campi vuoti in documenti > 5000 caratteri → probabile errore di estrazione
