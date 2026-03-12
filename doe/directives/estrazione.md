# Direttive di Estrazione per Campo

## oggetto_appalto
- Cerca dopo: "OGGETTO", "OGGETTO DELL'AFFIDAMENTO", "OGGETTO DELL'APPALTO"
- Di solito una frase lunga che descrive i lavori/servizi
- Può estendersi su più righe
- Ignora sottotitoli tipo "CIG:", "CPV:" che seguono

## stazione_appaltante
- Cerca in: intestazione, "STAZIONE APPALTANTE", "ENTE APPALTANTE", "COMMITTENTE"
- Nome completo dell'ente pubblico (Comune, Provincia, ASL, Ministero...)
- Potrebbe essere nel logo/header del documento

## responsabile_procedimento
- Cerca: "RUP", "Responsabile Unico del Procedimento", "Responsabile del Procedimento"
- È un nome e cognome, a volte con titolo (Ing., Dott., Arch.)

## cig
- Formato: 10 caratteri alfanumerici (es. "A1234B5678")
- Cerca: "CIG", "Codice Identificativo Gara"
- Se ci sono più lotti, ogni lotto ha il suo CIG

## cup
- Formato: 15 caratteri alfanumerici, inizia con lettera (es. "B12C34000560007")
- Cerca: "CUP", "Codice Unico Progetto"
- Non tutti i bandi hanno il CUP

## importo_base_asta
- Cerca: "importo a base di gara", "importo a base d'asta", "importo complessivo"
- Formato numerico senza simboli: 150000.00
- Attenzione: distingui dall'importo comprensivo di oneri sicurezza
- L'importo base è SENZA oneri sicurezza e IVA

## oneri_sicurezza
- Cerca: "oneri per la sicurezza", "oneri sicurezza non soggetti a ribasso"
- Sono NON soggetti a ribasso
- Formato numerico

## tipo_procedura
- Cerca: "PROCEDURA", "TIPO DI GARA"
- Valori comuni: "Procedura aperta", "Procedura negoziata", "Affidamento diretto",
  "Procedura ristretta", "Dialogo competitivo"

## criterio_aggiudicazione
- Cerca: "CRITERIO DI AGGIUDICAZIONE", "CRITERIO DI VALUTAZIONE"
- Valori: "Offerta economicamente più vantaggiosa (OEPV)",
  "Minor prezzo", "Costo fisso"
- Se OEPV: cercare rapporto punteggio tecnico/economico (es. 70/30, 80/20)

## durata_contratto
- Cerca: "DURATA", "TERMINE DI ESECUZIONE", "DURATA DEL CONTRATTO"
- Formato: "X mesi" o "X giorni" o data specifica
- Distingui tra durata contratto e tempo utile per completamento lavori

## garanzia_provvisoria
- Cerca: "GARANZIA PROVVISORIA", "CAUZIONE PROVVISORIA"
- Solitamente 2% dell'importo base
- Indicare importo e/o percentuale

## subappalto
- Cerca: "SUBAPPALTO", "SUBAPPALTARE"
- Indicare se ammesso (sì/no) e eventuali limiti percentuali
- Riferimento normativo: art. 119 D.Lgs. 36/2023

## sopralluogo
- Cerca: "SOPRALLUOGO"
- Indicare se obbligatorio e modalità di prenotazione
