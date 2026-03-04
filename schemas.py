"""
AppaltoAI — Pydantic Schema Enforcement + Coherence Validation
Modelli strutturati per validazione rigida dell'output di estrazione.
"""

from datetime import date, datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, model_validator
import re


# ═════════════════════════════════════════════════════════════════════════════
# MODELLI PYDANTIC — Estrazione Appalto
# ═════════════════════════════════════════════════════════════════════════════

class PageSource(BaseModel):
    """Sorgente pagina/sezione per un campo estratto."""
    field: str
    page: Optional[int] = None
    section: Optional[str] = None
    snippet: Optional[str] = None
    confidence: float = 0.0
    method: str = "unknown"


class CriterioOEPV(BaseModel):
    """Singolo criterio di valutazione OEPV."""
    descrizione: str = ""
    punteggio_max: Optional[float] = None
    tipo: str = ""  # "tecnico", "economico", "tabellare", "discrezionale"
    sub_criteri: List[dict] = Field(default_factory=list)


class AppaltoResult(BaseModel):
    """Schema completo per un appalto pubblico italiano.
    Tutti i campi con validazione integrata."""

    # ── Identificativi ────────────────────────────────────────────────
    cig: Optional[str] = Field(None, description="Codice Identificativo Gara (10 alfanumerico)")
    cup: Optional[str] = Field(None, description="Codice Unico Progetto")
    numero_gara: Optional[str] = Field(None, description="Numero di gara ANAC")

    # ── Stazione Appaltante ───────────────────────────────────────────
    stazione_appaltante: Optional[str] = None
    responsabile_procedimento: Optional[str] = Field(None, alias="rup")
    rup: Optional[str] = None

    # ── Oggetto ────────────────────────────────────────────────────────
    oggetto: Optional[str] = None
    tipo_appalto: Optional[str] = None  # "lavori", "servizi", "forniture"
    cpv: Optional[str] = None
    luogo_esecuzione: Optional[str] = None

    # ── Importi ────────────────────────────────────────────────────────
    importo_base_asta: Optional[str] = None
    importo_sicurezza: Optional[str] = None
    importo_totale: Optional[str] = None
    valore_stimato: Optional[str] = None
    oneri_sicurezza: Optional[str] = None

    # ── Date ───────────────────────────────────────────────────────────
    scadenza_offerte: Optional[str] = None
    data_pubblicazione: Optional[str] = None
    data_seduta_pubblica: Optional[str] = None
    durata_contratto: Optional[str] = None

    # ── Procedura ──────────────────────────────────────────────────────
    tipo_procedura: Optional[str] = None  # "aperta", "ristretta", "negoziata"
    criterio_aggiudicazione: Optional[str] = None  # "OEPV", "prezzo più basso"
    base_giuridica: Optional[str] = None
    ammissione_rti: Optional[str] = None

    # ── Punteggi ───────────────────────────────────────────────────────
    punteggio_tecnico: Optional[str] = None
    punteggio_economico: Optional[str] = None
    punteggio_totale: Optional[str] = None

    # ── Garanzie e requisiti ───────────────────────────────────────────
    garanzia_provvisoria: Optional[str] = None
    garanzia_definitiva: Optional[str] = None
    requisiti_capacita_tecnica: Optional[str] = None
    requisiti_capacita_economica: Optional[str] = None
    requisiti_soa: Optional[str] = None

    # ── Subappalto ─────────────────────────────────────────────────────
    subappalto: Optional[str] = None
    limite_subappalto: Optional[str] = None

    # ── OEPV criteri strutturati ───────────────────────────────────────
    criteri_oepv: List[CriterioOEPV] = Field(default_factory=list)

    # ── Metadati qualità ───────────────────────────────────────────────
    page_sources: List[PageSource] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    coherence_score: float = Field(0.0, ge=0.0, le=1.0)

    class Config:
        populate_by_name = True

    # ── Validatori CIG ─────────────────────────────────────────────────
    @field_validator("cig")
    @classmethod
    def validate_cig(cls, v):
        if v and not re.match(r'^[A-Z0-9]{10}$', v.upper().strip()):
            # Tenta pulizia
            cleaned = re.sub(r'[^A-Za-z0-9]', '', v.strip())
            if len(cleaned) == 10:
                return cleaned.upper()
            return None  # CIG non valido → rimuovi
        return v.upper().strip() if v else v

    # ── Validatori importi ─────────────────────────────────────────────
    @field_validator("importo_base_asta", "importo_sicurezza", "importo_totale",
                     "valore_stimato", "oneri_sicurezza", mode="before")
    @classmethod
    def validate_importi(cls, v):
        if v is None:
            return v
        v = str(v).strip()
        # Pulizia: rimuovi "Euro", "€", spazi
        cleaned = re.sub(r'[€Ee][Uu][Rr][Oo]|EUR|€', '', v).strip()
        cleaned = cleaned.replace(" ", "")
        # Accetta formato italiano (1.234.567,89) o internazionale
        if re.match(r'^[\d.,]+$', cleaned):
            return v
        return v

    # ── Validatori punteggi ────────────────────────────────────────────
    @field_validator("punteggio_tecnico", "punteggio_economico", "punteggio_totale", mode="before")
    @classmethod
    def validate_punteggi(cls, v):
        if v is None:
            return v
        # Prova a convertire in numero
        v_str = str(v).strip()
        cleaned = v_str.replace(",", ".")
        cleaned = re.sub(r'[^\d.]', '', cleaned)
        try:
            num = float(cleaned)
            if num < 0 or num > 200:  # Range ragionevole
                return None
            return v_str
        except ValueError:
            return v_str

    # ── Validatori date ────────────────────────────────────────────────
    @field_validator("scadenza_offerte", "data_pubblicazione", "data_seduta_pubblica", mode="before")
    @classmethod
    def validate_date(cls, v):
        if v is None:
            return v
        v = str(v).strip()
        # Accetta vari formati data italiani
        for pattern in [
            r'\d{2}/\d{2}/\d{4}',
            r'\d{2}\.\d{2}\.\d{4}',
            r'\d{2}-\d{2}-\d{4}',
            r'\d{1,2}\s+\w+\s+\d{4}',
        ]:
            if re.search(pattern, v):
                return v
        return v  # Conserva anche se formato non standard

    # ── Model-level cross-validation ───────────────────────────────────
    @model_validator(mode="after")
    def cross_validate(self):
        """Validazione incrociata tra campi correlati."""
        warnings = list(self.warnings) if self.warnings else []

        # Check: punteggio_totale ≈ tecnico + economico
        try:
            pt = _parse_number(self.punteggio_tecnico)
            pe = _parse_number(self.punteggio_economico)
            tot = _parse_number(self.punteggio_totale)
            if pt and pe and tot:
                if abs((pt + pe) - tot) > 1.0:
                    warnings.append(
                        f"⚠️ Incoerenza punteggi: tecnico({pt}) + economico({pe}) = {pt+pe} ≠ totale({tot})"
                    )
        except Exception:
            pass

        # Check: oneri sicurezza < importo base
        try:
            iba = _parse_money(self.importo_base_asta)
            ons = _parse_money(self.oneri_sicurezza)
            if iba and ons and ons > iba:
                warnings.append(
                    f"⚠️ Oneri sicurezza ({ons}) superiori all'importo base ({iba})"
                )
        except Exception:
            pass

        # Check: scadenza_offerte dopo data_pubblicazione
        try:
            dp = _parse_date_it(self.data_pubblicazione)
            so = _parse_date_it(self.scadenza_offerte)
            if dp and so and so < dp:
                warnings.append(
                    f"⚠️ Scadenza offerte ({self.scadenza_offerte}) precedente alla pubblicazione ({self.data_pubblicazione})"
                )
        except Exception:
            pass

        # Check: criterio OEPV ma nessun punteggio
        if self.criterio_aggiudicazione and "oepv" in self.criterio_aggiudicazione.lower():
            if not self.punteggio_tecnico and not self.punteggio_economico:
                warnings.append(
                    "⚠️ Criterio OEPV dichiarato ma nessun punteggio tecnico/economico trovato"
                )

        # Check: CIG presente
        if not self.cig:
            warnings.append("⚠️ CIG non trovato nel documento")

        self.warnings = warnings
        return self


# ═════════════════════════════════════════════════════════════════════════════
# COHERENCE VALIDATION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class CoherenceValidator:
    """Motore di validazione della coerenza dei dati estratti.
    Controlla relazioni logiche tra campi."""

    def __init__(self):
        self.checks = []
        self.score = 1.0

    def validate(self, result: AppaltoResult) -> dict:
        """Esegue tutti i controlli di coerenza. Ritorna report dettagliato."""
        self.checks = []
        self.score = 1.0
        penalty_per_issue = 0.1

        # 1. Punteggi coerenti
        self._check_punteggi(result)

        # 2. Importi coerenti
        self._check_importi(result)

        # 3. Date ordinate
        self._check_date(result)

        # 4. Criteri OEPV sum
        self._check_criteri_oepv(result)

        # 5. Completezza campi critici
        self._check_completeness(result)

        # 6. Formato valori
        self._check_formats(result)

        # Calcola score
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

    def _check_punteggi(self, r: AppaltoResult):
        pt = _parse_number(r.punteggio_tecnico)
        pe = _parse_number(r.punteggio_economico)
        tot = _parse_number(r.punteggio_totale)

        if pt is not None and pe is not None and tot is not None:
            expected = pt + pe
            if abs(expected - tot) <= 1.0:
                self._add("punteggi_sum", "pass", f"Tecnico({pt})+Economico({pe})={expected} ≈ Totale({tot})")
            else:
                self._add("punteggi_sum", "fail", f"Tecnico({pt})+Economico({pe})={expected} ≠ Totale({tot})")

        if pt is not None and (pt < 0 or pt > 100):
            self._add("punteggio_tecnico_range", "fail", f"Punteggio tecnico fuori range: {pt}")
        elif pt is not None:
            self._add("punteggio_tecnico_range", "pass", f"Punteggio tecnico {pt} nel range 0-100")

        if pe is not None and (pe < 0 or pe > 100):
            self._add("punteggio_economico_range", "fail", f"Punteggio economico fuori range: {pe}")
        elif pe is not None:
            self._add("punteggio_economico_range", "pass", f"Punteggio economico {pe} nel range 0-100")

    def _check_importi(self, r: AppaltoResult):
        iba = _parse_money(r.importo_base_asta)
        ons = _parse_money(r.oneri_sicurezza)
        itot = _parse_money(r.importo_totale)

        if iba is not None and iba > 0:
            self._add("importo_base_positivo", "pass", f"Importo base asta: {iba:,.2f}")
        elif iba is not None:
            self._add("importo_base_positivo", "fail", f"Importo base asta non positivo: {iba}")

        if iba and ons and ons > iba:
            self._add("oneri_vs_importo", "fail", f"Oneri sicurezza ({ons:,.2f}) > importo base ({iba:,.2f})")
        elif iba and ons:
            self._add("oneri_vs_importo", "pass", f"Oneri sicurezza ({ons:,.2f}) < importo base ({iba:,.2f})")

        if itot and iba and itot < iba * 0.5:
            self._add("importo_totale_vs_base", "warn",
                       f"Importo totale ({itot:,.2f}) molto inferiore a base asta ({iba:,.2f})")

    def _check_date(self, r: AppaltoResult):
        dp = _parse_date_it(r.data_pubblicazione)
        so = _parse_date_it(r.scadenza_offerte)
        ds = _parse_date_it(r.data_seduta_pubblica)

        if dp and so:
            if so >= dp:
                self._add("date_ordine_pub_scad", "pass", f"Pubblicazione({dp}) → Scadenza({so})")
            else:
                self._add("date_ordine_pub_scad", "fail", f"Scadenza({so}) prima di pubblicazione({dp})")

        if so and ds:
            if ds >= so:
                self._add("date_ordine_scad_seduta", "pass", f"Scadenza({so}) → Seduta({ds})")
            else:
                self._add("date_ordine_scad_seduta", "warn", f"Seduta({ds}) prima di scadenza({so})")

    def _check_criteri_oepv(self, r: AppaltoResult):
        if not r.criteri_oepv:
            return
        total_peso = sum(c.punteggio_max or 0 for c in r.criteri_oepv)
        tot = _parse_number(r.punteggio_totale) or 100
        if abs(total_peso - tot) <= 2:
            self._add("oepv_sum", "pass", f"Somma criteri OEPV ({total_peso}) ≈ totale ({tot})")
        elif total_peso > 0:
            self._add("oepv_sum", "fail", f"Somma criteri OEPV ({total_peso}) ≠ totale ({tot})")

    def _check_completeness(self, r: AppaltoResult):
        critical = ["cig", "oggetto", "stazione_appaltante", "importo_base_asta", "scadenza_offerte"]
        found = sum(1 for f in critical if getattr(r, f, None))
        total = len(critical)
        if found == total:
            self._add("completeness", "pass", f"Tutti i {total} campi critici presenti")
        elif found >= total * 0.6:
            missing = [f for f in critical if not getattr(r, f, None)]
            self._add("completeness", "warn", f"Campi critici mancanti: {', '.join(missing)}")
        else:
            missing = [f for f in critical if not getattr(r, f, None)]
            self._add("completeness", "fail", f"Troppi campi critici mancanti: {', '.join(missing)}")

    def _check_formats(self, r: AppaltoResult):
        if r.cig and not re.match(r'^[A-Z0-9]{10}$', r.cig.upper()):
            self._add("cig_format", "fail", f"CIG malformato: {r.cig}")
        elif r.cig:
            self._add("cig_format", "pass", f"CIG valido: {r.cig}")

        if r.cup and not re.match(r'^[A-Z]\d{2}[A-Z]\d{8}$', r.cup.upper().replace(" ", "")):
            self._add("cup_format", "warn", f"CUP potenzialmente malformato: {r.cup}")
        elif r.cup:
            self._add("cup_format", "pass", f"CUP valido: {r.cup}")


# ═════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def _parse_number(val: Optional[str]) -> Optional[float]:
    """Parsa un numero da stringa, gestendo formato italiano."""
    if val is None:
        return None
    v = str(val).strip()
    v = re.sub(r'[^\d.,\-]', '', v)
    if not v:
        return None
    # Formato italiano: 1.234,56 → 1234.56
    if ',' in v and '.' in v:
        v = v.replace('.', '').replace(',', '.')
    elif ',' in v:
        v = v.replace(',', '.')
    try:
        return float(v)
    except ValueError:
        return None


def _parse_money(val: Optional[str]) -> Optional[float]:
    """Parsa un importo monetario."""
    if val is None:
        return None
    v = str(val).strip()
    v = re.sub(r'[€Ee][Uu][Rr][Oo]|EUR|€|\s', '', v)
    return _parse_number(v)


def _parse_date_it(val: Optional[str]) -> Optional[date]:
    """Parsa una data italiana in vari formati."""
    if val is None:
        return None
    v = str(val).strip()
    # dd/mm/yyyy
    m = re.search(r'(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})', v)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass
    # dd mese yyyy (italiano)
    mesi = {
        'gennaio': 1, 'febbraio': 2, 'marzo': 3, 'aprile': 4,
        'maggio': 5, 'giugno': 6, 'luglio': 7, 'agosto': 8,
        'settembre': 9, 'ottobre': 10, 'novembre': 11, 'dicembre': 12,
    }
    m = re.search(r'(\d{1,2})\s+(\w+)\s+(\d{4})', v.lower())
    if m:
        mese = mesi.get(m.group(2))
        if mese:
            try:
                return date(int(m.group(3)), mese, int(m.group(1)))
            except ValueError:
                pass
    return None


def validate_extraction(raw_result: dict) -> AppaltoResult:
    """Converte un dict di estrazione grezza in un AppaltoResult validato."""
    # Gestisci alias campi
    field_aliases = {
        "responsabile_procedimento": "rup",
        "importo_a_base_di_gara": "importo_base_asta",
        "importo_a_base_d_asta": "importo_base_asta",
        "importo_gara": "importo_base_asta",
        "data_scadenza": "scadenza_offerte",
        "scadenza": "scadenza_offerte",
        "termine_offerte": "scadenza_offerte",
    }

    cleaned = {}
    for k, v in raw_result.items():
        # Normalizza chiave
        k_norm = k.lower().strip().replace(" ", "_")
        k_canon = field_aliases.get(k_norm, k_norm)
        if v and str(v).strip() and str(v).strip().lower() not in ("n/a", "non trovato", "-", ""):
            cleaned[k_canon] = str(v).strip()

    try:
        result = AppaltoResult(**cleaned)
    except Exception:
        # Se Pydantic fallisce, crea con i campi che matchano
        valid_fields = set(AppaltoResult.model_fields.keys())
        safe = {k: v for k, v in cleaned.items() if k in valid_fields}
        result = AppaltoResult(**safe)
        result.warnings.append("⚠️ Alcuni campi estratti non corrispondono allo schema atteso")

    return result


def full_validation(raw_result: dict) -> dict:
    """Pipeline completa: schema validation + coherence check.
    Ritorna il risultato validato + report coerenza."""
    validated = validate_extraction(raw_result)
    validator = CoherenceValidator()
    coherence = validator.validate(validated)
    validated.coherence_score = coherence["coherence_score"]

    return {
        "result": validated.model_dump(exclude_none=True),
        "coherence": coherence,
        "warnings": validated.warnings,
    }
