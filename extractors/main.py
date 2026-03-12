"""
Orchestrator – assembla tutte le sezioni di estrazione e fornisce le API pubbliche.

Funzioni pubbliche (ri-esportate anche da __init__):

* extract_rules_based(text) -> dict
* flatten_for_pipeline(nested) -> (flat, snippets, methods)
* extract_from_pdf_bytes(pdf_bytes, filename) -> (flat, snippets, methods, text)
* extract_from_text_direct(text) -> (flat, snippets, methods)
* extract_disciplinare(pdf_path, provider, model, save_output) -> dict
* extract_all_disciplinari(folder, provider, model) -> list[dict]
"""

from __future__ import annotations
import json, os, re
from pathlib import Path
from typing import Optional

# Sotto-moduli di estrazione
from .pdf import extract_text_from_pdf
from .llm import build_extraction_prompt, call_openai, call_anthropic
from .info_generali import extract_info_generali
from .procedura import extract_procedura
from .piattaforma import extract_piattaforma
from .lotti import extract_lotti
from .tempistiche import extract_tempistiche
from .requisiti import extract_requisiti
from .valutazione import extract_valutazione
from .offerta import extract_offerta
from .garanzie import extract_garanzie
from .complementari import extract_complementari
from .flatten import flatten_for_pipeline


# ──────────────────────────────────────────────────────────────────────────
# Pulizia finale (identica alla versione monolitica)
# ──────────────────────────────────────────────────────────────────────────
def _clean_empty(d):
    if isinstance(d, dict):
        cleaned = {}
        for k, v in d.items():
            cv2 = _clean_empty(v)
            if cv2 is not None and cv2 != {} and cv2 != [] and cv2 != "":
                cleaned[k] = cv2
        return cleaned if cleaned else None
    elif isinstance(d, list):
        cleaned = [_clean_empty(item) for item in d if _clean_empty(item) is not None]
        return cleaned if cleaned else None
    else:
        return d


# ──────────────────────────────────────────────────────────────────────────
# Funzione principale di estrazione basata su regole
# ──────────────────────────────────────────────────────────────────────────
def extract_rules_based(text: str) -> dict:
    """
    Estrazione COMPLETA basata su regole/regex.
    Copre ~60+ campi dello schema usando pattern matching + estrazione per sezioni.
    """
    text_lower = text.lower()

    # A) Informazioni generali
    ig = extract_info_generali(text, text_lower)

    # B) Tipo procedura
    tp = extract_procedura(text, text_lower)

    # C) Piattaforma telematica
    pt = extract_piattaforma(text, text_lower)

    # D) Lotti e importi (modifica ig in-place per categorie_trovate)
    sl, ic = extract_lotti(text, text_lower, ig)

    # E) Tempistiche
    temp, durata_contratto = extract_tempistiche(text, text_lower)

    # F) Requisiti di partecipazione
    rp = extract_requisiti(text, text_lower)

    # G) Criteri di valutazione
    cv = extract_valutazione(text, text_lower)

    # H + H-bis) Formato offerta tecnica
    otf = extract_offerta(text, text_lower)

    # I) Garanzie
    gar = extract_garanzie(text, text_lower)

    # J-U) Sezioni complementari
    comp = extract_complementari(text, text_lower)

    # ── Assemblaggio risultato ───────────────────────────────────────────
    result: dict = {
        "informazioni_generali": ig,
        "tipo_procedura": tp,
        "piattaforma_telematica": pt,
        "suddivisione_lotti": sl,
        "importi_complessivi": ic,
        "tempistiche": temp,
        "requisiti_partecipazione": rp,
        "criteri_valutazione": cv,
        "offerta_tecnica_formato": otf,
        "garanzie": gar,
    }

    # Durata contratto (top-level, da tempistiche)
    if durata_contratto:
        result["durata_contratto"] = durata_contratto

    # Merge delle sezioni complementari (J-U)
    result.update(comp)

    # Pulizia finale: rimuovi sezioni vuote
    result = _clean_empty(result) or {}
    return result


# ──────────────────────────────────────────────────────────────────────────
# API pubbliche
# ──────────────────────────────────────────────────────────────────────────
def extract_from_pdf_bytes(pdf_bytes: bytes, filename: str = "upload.pdf") -> tuple[dict, dict, dict, str]:
    """
    Estrae dati da bytes PDF e ritorna nel formato pipeline.

    Returns:
        (flat_result, snippets, methods, text)
    """
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        text = extract_text_from_pdf(tmp_path)
        nested = extract_rules_based(text)
        flat, snippets, methods = flatten_for_pipeline(nested)
        return flat, snippets, methods, text
    finally:
        os.unlink(tmp_path)


def extract_from_text_direct(text: str) -> tuple[dict, dict, dict]:
    """
    Estrae dati da testo e ritorna nel formato pipeline.

    Returns:
        (flat_result, snippets, methods)
    """
    nested = extract_rules_based(text)
    flat, snippets, methods = flatten_for_pipeline(nested)
    return flat, snippets, methods


def extract_disciplinare(
    pdf_path: str,
    provider: str = "rules",
    model: Optional[str] = None,
    save_output: bool = True,
) -> dict:
    """Pipeline completa di estrazione dati da un disciplinare."""
    print(f"[PDF] Estrazione da: {os.path.basename(pdf_path)}")

    print("  > Estrazione testo dal PDF...")
    pdf_text = extract_text_from_pdf(pdf_path)
    print(f"  > Estratti {len(pdf_text):,} caratteri")

    if provider == "rules":
        print("  > Estrazione con regole (senza LLM)...")
        result = extract_rules_based(pdf_text)
    else:
        print(f"  > Costruzione prompt per {provider}...")
        messages = build_extraction_prompt(pdf_text)
        print(f"  > Chiamata {provider} ({model or 'default'})...")
        if provider == "openai":
            raw = call_openai(messages, model=model or "gpt-4o")
        elif provider == "anthropic":
            raw = call_anthropic(messages, model=model or "claude-sonnet-4-20250514")
        else:
            raise ValueError(f"Provider non supportato: {provider}")
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {"errore": "Impossibile parsare la risposta", "raw": raw}

    if save_output:
        out_path = Path(pdf_path).with_suffix(".extracted.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  [OK] Salvato: {out_path.name}")

    return result


def extract_all_disciplinari(
    folder: str = ".",
    provider: str = "rules",
    model: Optional[str] = None,
) -> list[dict]:
    """Estrae dati da TUTTI i PDF disciplinari in una cartella."""
    pdf_files = sorted(Path(folder).glob("*.pdf"))
    disciplinari = [
        p for p in pdf_files
        if any(kw in p.name.lower() for kw in ["disciplinare", "disciplinar", "bando"])
    ]
    if not disciplinari:
        disciplinari = pdf_files

    print(f"\n{'='*60}")
    print(f"  Estrazione batch: {len(disciplinari)} disciplinari")
    print(f"  Provider: {provider}")
    print(f"{'='*60}\n")

    results = []
    for i, pdf_path in enumerate(disciplinari, 1):
        print(f"\n[{i}/{len(disciplinari)}] {pdf_path.name}")
        print("-" * 50)
        try:
            result = extract_disciplinare(str(pdf_path), provider=provider, model=model)
            result["_source_file"] = pdf_path.name
            results.append(result)
        except Exception as e:
            print(f"  [ERR] ERRORE: {e}")
            results.append({"_source_file": pdf_path.name, "_error": str(e)})

    summary_path = Path(folder) / "estrazione_riepilogo.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Riepilogo salvato: {summary_path}")

    return results


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Estrai dati strutturati da disciplinari di gara"
    )
    parser.add_argument("input", nargs="?", default=".",
                        help="Percorso PDF singolo o cartella")
    parser.add_argument("--provider", choices=["openai", "anthropic", "rules"],
                        default="rules")
    parser.add_argument("--model", default=None)
    parser.add_argument("--all", action="store_true")

    args = parser.parse_args()

    if args.all or os.path.isdir(args.input):
        folder = args.input if os.path.isdir(args.input) else "."
        extract_all_disciplinari(folder, provider=args.provider, model=args.model)
    else:
        extract_disciplinare(args.input, provider=args.provider, model=args.model)
