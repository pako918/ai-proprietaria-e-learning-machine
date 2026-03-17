import re, sys
from extractors.procedura import extract_procedura

# Test 1: procedura
text_proc = "Procedura negoziata senza bando, ai sensi dell'art. 50 comma 1 lett. d) del d.lgs. 36/2023. Il Responsabile della Sicurezza art. 98 del D.lgs. 81/2008."
tp = extract_procedura(text_proc, text_proc.lower())
print("=== Test procedura ===")
print("tipo:", tp.get("tipo"))
print("senza_bando:", tp.get("senza_bando"))
print("rif_norm:", tp.get("riferimento_normativo"))
print()

# Test 2: lotti
from extractors.lotti import extract_lotti
text_lotti = (
    "LOTTO 1 DESCRIZIONE LAVORI NAPOLI\n"
    "Importo massimo pagabile euro 663.428,35 di cui 555.465,01 soggetti a ribasso e "
    "18.429,21 per oneri di sicurezza non soggetti a ribasso. Ulteriori dettagli sul "
    "contratto e sulle modalita di stipula. Il contratto di appalto e: a misura per la "
    "quota lavori e a corpo per la quota progettazione esecutiva. Luogo Napoli.\n"
    "LOTTO 2 DESCRIZIONE LAVORI NAPOLI\n"
    "Importo massimo pagabile euro 250.318,67 di cui 235.000,00 soggetti a ribasso e "
    "15.318,67 per oneri di sicurezza non soggetti a ribasso. Ulteriori dettagli sul "
    "contratto e sulle modalita di stipula per il secondo lotto. Luogo Napoli.\n"
    "LOTTO 3 DESCRIZIONE LAVORI NAPOLI\n"
    "Importo massimo pagabile euro 709.242,04. Durata dei lavori 365 giorni.\n"
)
ig = {}
sl, ic = extract_lotti(text_lotti, text_lotti.lower(), ig)
print("=== Test lotti ===")
for l in sl.get("lotti", []):
    print("  Lotto", l["numero"], "importo=", l.get("importo_base_asta"),
          "oneri=", l.get("oneri_sicurezza_non_ribassabili"))
print("tipo_contratto:", ic.get("tipo_contratto"))
print()

# Test 3: json_builder tipologia + note
from json_builder import build_output
nested = {
    "tipo_procedura": {
        "tipo": "negoziata",
        "senza_bando": True,
        "riferimento_normativo": "art. 50 comma 1 lett. d) del d.lgs. 36/2023",
        "criterio_aggiudicazione": "OEPV",
    },
    "suddivisione_lotti": {"lotti": []},
    "importi_complessivi": {
        "importo_totale_gara": 1483423.01,
        "tipo_contratto": "Il contratto e: a misura per la quota lavori.",
    },
    "informazioni_generali": {},
    "piattaforma_telematica": {},
    "tempistiche": {},
    "requisiti_partecipazione": {},
    "criteri_valutazione": {},
    "garanzie": {},
    "sopralluogo": {},
    "informazioni_aggiuntive": {},
    "documentazione_amministrativa": {},
    "aggiudicazione": {},
    "offerta_tecnica_formato": {},
}
out = build_output(nested)
dl = out.get("descrizione_lavori_con_importo_totale") or {}
print("=== Test json_builder ===")
print("tipologia:", dl.get("tipologia"))
print("note:", dl.get("note"))
print()

# Test 4: cat_matches lookbehind
print("=== Test cat codes ===")
sample = "OS30 Impianti elettrici 3,50 15.000,00\nOS28 Scavi 2,00 8.000,00\nE.20 Progettazione impianti 1,50 6.000,00\nIA.03 Installazioni 2,25 12.000,00\nS.28 Idraulica 1,75 9.000,00"
cat_matches = re.findall(
    r"(?<![A-Z0-9])(E\.?\d{2}|S\.?\d{2}|IA\.?\d{2}|D\.?\d{2}|V\.?\d{2})\s+"
    r"([^\n]{5,200}?)\s+(\d{1,2}[.,]\d{2})\s+([\d.]+[.,]\d{2}(?:\s*\u20ac)?)",
    sample,
)
print("cat_matches:", cat_matches)
print()

# Test 5: periodo requisiti
from extractors.requisiti import extract_requisiti
text_req = (
    "Requisiti di capacita tecnica e professionale. "
    "Il concorrente deve aver regolarmente eseguito, nei dieci anni antecedenti alle date indicate nel bando di gara, "
    "servizi di ingegneria per importo non inferiore a euro 500.000,00."
)
r = extract_requisiti(text_req, text_req.lower())
print("=== Test periodo ===")
sa = (r.get("capacita_tecnico_professionale") or {}).get("servizi_analoghi") or {}
print("  periodo:", sa.get("periodo_riferimento"))
