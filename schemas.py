"""
AppaltoAI — Validazione Schema + Coerenza
Validazione dell'output strutturato basato su output_schema.py.
"""

from datetime import date
from typing import Optional, Dict, Any, List
from pydantic import ValidationError
import re

from output_schema import AppaltoOutput


# ═════════════════════════════════════════════════════════════════════════════
# COHERENCE VALIDATION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class CoherenceValidator:
    """Motore di validazione della coerenza dei dati estratti.
    Opera sulla struttura nested (output_schema)."""

    def __init__(self):
        self.checks = []
        self.score = 1.0

    def validate(self, data: dict) -> dict:
        """Esegue tutti i controlli di coerenza su un dict output.
        Ritorna report dettagliato."""
        self.checks = []
        self.score = 1.0
        penalty_per_issue = 0.1

        self._check_punteggi(data)
        self._check_importi(data)
        self._check_criteri_sum(data)
        self._check_completeness(data)
        self._check_formats(data)

        n_issues = sum(1 for c in self.checks if c["status"] == "fail")
        self.score = max(0.0, 1.0 - n_issues * penalty_per_issue)

        return {
            "coherence_score": round(self.score, 2),
            "total_checks": len(self.checks),
            "passed": sum(1 for c in self.checks if c["status"] == "pass"),
            "warnings": sum(1 for c in self.checks if c["status"] == "warn"),
            "failures": n_issues,
            "details": self.checks,
        }

    def _add(self, name: str, status: str, message: str):
        self.checks.append({"check": name, "status": status, "message": message})

    def _check_punteggi(self, d: dict):
        """Verifica coerenza punteggi tecnica + economica = totale."""
        ot = d.get("offerta_tecnica", {}) or {}
        oe = d.get("offerta_economica", {}) or {}
        crit = d.get("criteri_valutazione_offerta_tecnica", {}) or {}

        pt = ot.get("punteggio_massimo")
        pe = oe.get("punteggio_massimo")
        totale = crit.get("punteggio_totale")

        if pt is not None and pe is not None and totale is not None:
            expected = pt + pe
            if abs(expected - totale) <= 1.0:
                self._add("punteggi_sum", "pass",
                           f"Tecnico({pt})+Economico({pe})={expected} ≈ Totale({totale})")
            else:
                self._add("punteggi_sum", "fail",
                           f"Tecnico({pt})+Economico({pe})={expected} ≠ Totale({totale})")

        rip = crit.get("ripartizione", {}) or {}
        if rip.get("offerta_tecnica") is not None and pt is not None:
            if abs(rip["offerta_tecnica"] - pt) > 0.5:
                self._add("ripartizione_tecnica", "fail",
                           f"Ripartizione tecnica ({rip['offerta_tecnica']}) ≠ punteggio OT ({pt})")

        if pt is not None and (pt < 0 or pt > 100):
            self._add("punteggio_tecnico_range", "fail", f"Punteggio tecnico fuori range: {pt}")
        elif pt is not None:
            self._add("punteggio_tecnico_range", "pass", f"Punteggio tecnico {pt} nel range 0-100")

        if pe is not None and (pe < 0 or pe > 100):
            self._add("punteggio_economico_range", "fail", f"Punteggio economico fuori range: {pe}")
        elif pe is not None:
            self._add("punteggio_economico_range", "pass", f"Punteggio economico {pe} nel range 0-100")

    def _check_importi(self, d: dict):
        """Verifica coerenza importi lotti vs totale."""
        descr = d.get("descrizione_lavori_con_importo_totale", {}) or {}
        totale = descr.get("importo_totale_procedura_euro")
        lotti = descr.get("lotti", [])

        if lotti and totale:
            somma_lotti = sum(l.get("importo_euro", 0) or 0 for l in lotti)
            if somma_lotti > 0:
                if abs(somma_lotti - totale) <= totale * 0.02:
                    self._add("importi_lotti_sum", "pass",
                               f"Somma lotti ({somma_lotti:,.2f}) ≈ Totale ({totale:,.2f})")
                else:
                    self._add("importi_lotti_sum", "warn",
                               f"Somma lotti ({somma_lotti:,.2f}) ≠ Totale ({totale:,.2f})")

        if totale is not None and totale > 0:
            self._add("importo_totale_positivo", "pass", f"Importo totale: {totale:,.2f}")
        elif totale is not None:
            self._add("importo_totale_positivo", "fail", f"Importo totale non positivo: {totale}")

        # Quota fissa + ribassabile ≈ importo lotto
        for lotto in lotti:
            imp = lotto.get("importo_euro")
            qf = lotto.get("quota_fissa_65_percento_euro")
            qr = lotto.get("quota_ribassabile_35_percento_euro")
            n = lotto.get("lotto", "?")
            if imp and qf and qr:
                if abs((qf + qr) - imp) <= 1.0:
                    self._add(f"quote_lotto_{n}", "pass",
                               f"Lotto {n}: quota fissa + ribassabile = importo")
                else:
                    self._add(f"quote_lotto_{n}", "fail",
                               f"Lotto {n}: {qf}+{qr}={qf+qr} ≠ {imp}")

    def _check_criteri_sum(self, d: dict):
        """Verifica che la somma dei criteri = punteggio max OT."""
        ot = d.get("offerta_tecnica", {}) or {}
        criteri = ot.get("criteri", [])
        pt_max = ot.get("punteggio_massimo")

        if criteri and pt_max:
            somma_criteri = sum(c.get("punteggio", 0) or 0 for c in criteri)
            if abs(somma_criteri - pt_max) <= 1.0:
                self._add("criteri_sum", "pass",
                           f"Somma criteri ({somma_criteri}) ≈ max OT ({pt_max})")
            elif somma_criteri > 0:
                self._add("criteri_sum", "warn",
                           f"Somma criteri ({somma_criteri}) ≠ max OT ({pt_max})")

    def _check_completeness(self, d: dict):
        """Verifica presenza campi critici."""
        critical = {
            "cig": d.get("cig"),
            "oggetto_appalto": d.get("oggetto_appalto"),
            "stazione_appaltante": d.get("stazione_appaltante"),
            "scadenza": d.get("scadenza"),
            "tipologia_appalto": d.get("tipologia_appalto"),
        }
        found = sum(1 for v in critical.values() if v)
        total = len(critical)
        missing = [k for k, v in critical.items() if not v]

        if found == total:
            self._add("completeness", "pass", f"Tutti i {total} campi critici presenti")
        elif found >= total * 0.6:
            self._add("completeness", "warn", f"Campi critici mancanti: {', '.join(missing)}")
        else:
            self._add("completeness", "fail", f"Troppi campi critici mancanti: {', '.join(missing)}")

    def _check_formats(self, d: dict):
        """Verifica formato CIG e CUP."""
        cig = d.get("cig", {})
        if isinstance(cig, dict):
            for lotto_key, cig_val in cig.items():
                if cig_val and re.match(r'^[A-Z0-9]{10}$', str(cig_val).upper()):
                    self._add(f"cig_format_{lotto_key}", "pass", f"CIG {lotto_key} valido: {cig_val}")
                elif cig_val:
                    self._add(f"cig_format_{lotto_key}", "fail", f"CIG {lotto_key} malformato: {cig_val}")

        cup = d.get("cup")
        if cup and re.match(r'^[A-Z]\d{2}[A-Z]\d{8}\d*$', str(cup).upper().replace(" ", "")):
            self._add("cup_format", "pass", f"CUP valido: {cup}")
        elif cup:
            self._add("cup_format", "warn", f"CUP potenzialmente malformato: {cup}")


# ═════════════════════════════════════════════════════════════════════════════
# VALIDATION PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def validate_schema(raw_result: dict) -> tuple[dict, list]:
    """Valida un dict di output contro AppaltoOutput.
    Ritorna (cleaned_dict, warnings)."""
    warnings = []
    # Rimuovi chiavi interne (_*)
    data = {k: v for k, v in raw_result.items() if not k.startswith("_")}
    try:
        model = AppaltoOutput(**data)
        return model.model_dump(exclude_none=True), warnings
    except ValidationError as e:
        for err in e.errors():
            loc = " → ".join(str(x) for x in err["loc"])
            warnings.append(f"Schema: {loc}: {err['msg']}")
        # Ritorna il dict originale con i warning
        return data, warnings


def full_validation(raw_result: dict) -> dict:
    """Pipeline completa: schema validation + coherence check.
    Lavora con la struttura nested (output_schema)."""
    # Rimuovi chiavi interne per la validazione
    data = {k: v for k, v in raw_result.items() if not k.startswith("_")}

    # Schema validation
    cleaned, schema_warnings = validate_schema(data)

    # Coherence validation
    validator = CoherenceValidator()
    coherence = validator.validate(cleaned)

    # Combina warnings
    all_warnings = schema_warnings + [
        c["message"] for c in coherence["details"] if c["status"] in ("fail", "warn")
    ]

    return {
        "result": cleaned,
        "coherence": coherence,
        "warnings": all_warnings,
    }
