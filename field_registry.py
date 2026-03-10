"""
AppaltoAI — Registro Centralizzato dei Campi
Fonte unica di verità per campi di estrazione, validatori, pattern, categorie.
Supporta campi dinamici (custom) persistiti in SQLite.
"""

import re
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable, List, Dict, Any
from datetime import datetime

from config import DB_PATH
from database import get_connection, init_custom_fields_table
from log_config import get_logger

logger = get_logger("field_registry")


# ═══════════════════════════════════════════════════════════════════════════
# DEFINIZIONE CAMPO
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FieldDef:
    """Definizione completa di un campo estraibile."""
    key: str                          # ID univoco (es. 'cig', 'importo_totale')
    label: str                        # Label UI (es. 'CIG', 'Importo Totale')
    icon: str = "📋"                  # Emoji per UI
    category: str = "Altro"           # Sezione UI (es. 'Identificativi', 'Importi')
    field_type: str = "text"          # text | money | number | boolean | date | list | object
    mono: bool = False                # Font monospace in UI
    highlight: bool = False           # Evidenziato in UI
    full_width: bool = False          # Larghezza piena in UI
    patterns: List[str] = field(default_factory=list)     # Regex patterns per estrazione
    validator_type: str = ""          # Tipo validatore: cig, cup, cpv, importi, punteggi, date, text, nuts
    is_custom: bool = False           # True se aggiunto dall'utente
    auto_learn_blacklist: bool = False # Non apprende automaticamente
    description: str = ""             # Descrizione del campo
    extraction_hint: str = ""         # Suggerimento per l'estrazione

    def to_dict(self) -> dict:
        """Serializza per API/frontend."""
        return {
            "key": self.key,
            "label": self.label,
            "icon": self.icon,
            "category": self.category,
            "field_type": self.field_type,
            "mono": self.mono,
            "highlight": self.highlight,
            "full_width": self.full_width,
            "patterns": self.patterns,
            "validator_type": self.validator_type,
            "is_custom": self.is_custom,
            "auto_learn_blacklist": self.auto_learn_blacklist,
            "description": self.description,
            "extraction_hint": self.extraction_hint,
        }


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATORI
# ═══════════════════════════════════════════════════════════════════════════

def _validate_cig(v):
    return bool(re.match(r'^[A-Z0-9]{10}$', str(v).strip()))

def _validate_cup(v):
    return bool(re.match(r'^[A-Z][0-9]{2}[A-Z][0-9]{8}$', str(v).strip()))

def _validate_cpv(v):
    return bool(re.match(r'^\d{8}-\d$', str(v).strip()))

def _validate_nuts(v):
    return bool(re.match(r'^IT[A-Z0-9]{1,3}$', str(v).strip(), re.I))

def _validate_importi(v):
    try:
        n = float(str(v).replace('.', '').replace(',', '.').replace('€', '').strip())
        return 0 < n < 50_000_000_000
    except (ValueError, TypeError):
        return False

def _validate_punteggi(v):
    try:
        return 0 <= int(v) <= 100
    except (ValueError, TypeError):
        return False

def _validate_date(v):
    return bool(re.match(r'\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}', str(v)))

def _validate_text(v):
    s = str(v).strip()
    return 3 <= len(s) <= 2000

def _validate_boolean(v):
    return isinstance(v, bool)


VALIDATORS: Dict[str, Callable] = {
    "cig": _validate_cig,
    "cup": _validate_cup,
    "cpv": _validate_cpv,
    "nuts": _validate_nuts,
    "importi": _validate_importi,
    "punteggi": _validate_punteggi,
    "date": _validate_date,
    "text": _validate_text,
    "boolean": _validate_boolean,
}


def get_validator(validator_type: str) -> Optional[Callable]:
    """Restituisce il validatore per il tipo specificato."""
    return VALIDATORS.get(validator_type)


# ═══════════════════════════════════════════════════════════════════════════
# CAMPI BUILT-IN — definizioni complete
# ═══════════════════════════════════════════════════════════════════════════

BUILTIN_FIELDS: List[FieldDef] = [
    # ── Identificativi ─────────────────────────────────────────────
    FieldDef("cig", "CIG", "🔑", "Identificativi", "text", mono=True,
             patterns=[r'(?:CIG|C\.I\.G\.?)[\s:]*([A-Z0-9]{10})\b'],
             validator_type="cig",
             description="Codice Identificativo Gara"),
    FieldDef("cup", "CUP", "📌", "Identificativi", "text", mono=True,
             patterns=[r'(?:CUP|C\.U\.P\.?)[\s:]*([A-Z]\d{2}[A-Z]\d{8})\b'],
             validator_type="cup",
             description="Codice Unico Progetto"),
    FieldDef("cpv", "Codice CPV", "🗂", "Identificativi", "text", mono=True,
             patterns=[r'(?:CPV|C\.P\.V\.?)[\s:]*(\d{8}-\d)'],
             validator_type="cpv",
             description="Vocabolario comune appalti"),
    FieldDef("nuts_code", "Codice NUTS", "🌍", "Identificativi", "text", mono=True,
             patterns=[r'(?:NUTS|codice\s+NUTS)[\s:]*\(?([A-Z]{2}[A-Z0-9]{1,3})\)?'],
             validator_type="nuts"),
    FieldDef("codice_progetto", "Codice Progetto", "🔢", "Identificativi", "text", mono=True,
             patterns=[r'(?:codice\s+progetto|codice\s+intervento|CUI)[\s:]+([A-Z0-9\-/]{5,30})'],
             validator_type="text",
             description="Codice progetto/intervento/CUI"),

    # ── Soggetti ───────────────────────────────────────────────────
    FieldDef("stazione_appaltante", "Stazione Appaltante", "🏛", "Soggetti", "text",
             highlight=True,
             patterns=[
                 r'(?:stazione\s+appaltante|ente\s+appaltante|amministrazione\s+aggiudicatrice)[\s:/–\-]*\n?\s*([^\n]{5,120})',
                 r'Spett\.le\s+([^\n]{5,100})',
             ],
             validator_type="text"),
    FieldDef("amministrazione_delegante", "Amm. Delegante", "🏢", "Soggetti", "text",
             patterns=[r'(?:amministrazione\s+delegante|ente\s+delegante)[\s:/–\-]*\n?\s*([^\n]{5,120})'],
             validator_type="text"),
    FieldDef("rup", "RUP", "👤", "Soggetti", "text",
             patterns=[
                 r'(?:R\.?U\.?P\.?|Responsabile\s+Unico\s+(?:del\s+)?Procedimento)[\s:–\-]*\s*(?:(?:Ing|Arch|Dott|Geom|Prof|Avv)\.?\s+)?([A-Z][a-zà-ú]+(?:\s+[A-Z][a-zà-ú]+){1,3})',
             ],
             validator_type="text"),
    FieldDef("responsabile_procedimento", "Resp. Procedimento", "👤", "Soggetti", "text",
             patterns=[r'(?:Responsabile\s+del\s+Procedimento|DEC)[\s:–\-]*\s*(?:(?:Ing|Arch|Dott|Geom|Prof|Avv)\.?\s+)?([A-Z][a-zà-ú]+(?:\s+[A-Z][a-zà-ú]+){1,3})'],
             validator_type="text"),
    FieldDef("direttore_esecuzione", "Direttore Esecuzione", "👷", "Soggetti", "text",
             patterns=[r"(?:Direttore\s+(?:dell['’]?\s*)?(?:Esecuzione|Lavori)|D\.?L\.?)[\s:\u2013\-]*\s*(?:(?:Ing|Arch|Dott|Geom)\.?\s+)?([A-Z][a-z\u00e0-\u00fa]+(?:\s+[A-Z][a-z\u00e0-\u00fa]+){1,3})"],
             validator_type="text",
             description="Direttore Esecuzione/Lavori"),
    FieldDef("coordinatore_sicurezza", "Coordinatore Sicurezza", "⛑️", "Soggetti", "text",
             patterns=[r'(?:Coordinatore\s+(?:della?\s+)?Sicurezza|CSP|CSE)[\s:–\-]*\s*(?:(?:Ing|Arch|Dott|Geom)\.?\s+)?([A-Z][a-zà-ú]+(?:\s+[A-Z][a-zà-ú]+){1,3})'],
             validator_type="text"),

    # ── Oggetto e Procedura ────────────────────────────────────────
    FieldDef("oggetto_appalto", "Oggetto Appalto", "📋", "Oggetto e Procedura", "text",
             highlight=True, full_width=True,
             patterns=[
                 # Pattern label esplicita: "OGGETTO: ..."
                 r"(?:OGGETTO|Oggetto\s+dell['\u2019]?\s*appalto)[\s:\u2013\-]+\n?\s*([^\n]{10,500})",
                 # "avente per/ad oggetto ..."
                 r'avente\s+(?:per|ad)\s+oggetto[\s:]+([^\n]{10,500})',
                 # "PROCEDURA ... PER/RELATIVA ..." (titolo tipo intestazione)
                 r'PROCEDURA\s+(?:APERTA|NEGOZIATA|RISTRETTA|COMPETITIVA)\s+(?:PER|RELATIVA\s+A)\s+(?:IL\s+|LA\s+|L[\u2019\']|AL\s+|ALLA\s+)?(.+?)(?:\n(?:CIG|CUP|Pag\.|Art\.))',
                 # "DISCIPLINARE DI GARA ... per/relativo ..."
                 r'DISCIPLINARE\s+DI\s+GARA\s+(?:PER|RELATIV[OA]\s+A)\s+(?:IL\s+|LA\s+|L[\u2019\'])?(.+?)(?:\n(?:CIG|CUP|Pag\.))',
                 # "BANDO DI GARA PER ..."
                 r'BANDO\s+DI\s+GARA\s+(?:PER|RELATIV[OA]\s+A)\s+(?:IL\s+|LA\s+|L[\u2019\'])?(.+?)(?:\n(?:CIG|CUP|Pag\.))',
                 # "Oggetto:" su una riga, valore sulla successiva
                 r'(?:^|\n)\s*OGGETTO\s*[:\n]\s*\n\s*([^\n]{10,500})',
                 # "1. Oggetto dell'appalto" → paragrafo successivo
                 r'(?:\d+[.)\s]+)?(?:OGGETTO|Oggetto)\s+(?:DELL[\u2019\']?\s*)?(?:APPALTO|GARA|AFFIDAMENTO|SERVIZIO|INCARICO)[\s:.\-]*\n+\s*([^\n]{10,500})',
                 # "Appalto di/dei/per ..."
                 r'(?:^|\n)\s*APPALTO\s+(?:DEI|DI|PER)\s+(.+?)(?:\n(?:CIG|CUP|Pag\.))',
                 # "AFFIDAMENTO dell'incarico / del servizio / dei lavori di ..."
                 r'AFFIDAMENTO\s+(?:DELL[\u2019\']?\s*INCARICO|DEL\s+SERVIZIO|DEI\s+(?:LAVORI|SERVIZI))\s+(?:DI\s+|PER\s+)?(.+?)(?:\n(?:CIG|CUP|Pag\.))',
                 # "GARA ... PER ..."
                 r'GARA\s+(?:(?:EUROPEA|COMUNITARIA|PUBBLICA)\s+)?(?:PER|RELATIVA)\s+(?:A\s+)?(?:IL\s+|LA\s+|L[\u2019\'])?(.+?)(?:\n(?:CIG|CUP|Art\.))',
             ],
             validator_type="text"),
    FieldDef("tipo_procedura", "Tipo Procedura", "⚖️", "Oggetto e Procedura", "text",
             description="Classificata dal motore AI (aperta, negoziata, etc.)"),
    FieldDef("criterio_aggiudicazione", "Criterio Aggiudicazione", "🎯", "Oggetto e Procedura", "text",
             full_width=True,
             description="OEPV, prezzo più basso, etc."),
    FieldDef("is_accordo_quadro", "Accordo Quadro", "📑", "Oggetto e Procedura", "boolean"),
    FieldDef("tipo_accordo_quadro", "Tipo Accordo Quadro", "📑", "Oggetto e Procedura", "text",
             patterns=[r'accordo\s+quadro\s+(?:a|con)\s+(unico|doppio)'],
             validator_type="text"),
    FieldDef("numero_lotti", "Numero Lotti", "📊", "Oggetto e Procedura", "number"),
    FieldDef("luogo_esecuzione", "Luogo Esecuzione", "📍", "Oggetto e Procedura", "text",
             patterns=[
                 r'(?:luogo\s+(?:di\s+)?(?:esecuzione|consegna|prestazione))[\s:–\-]*\n?\s*([^\n]{5,150})',
                 r'(?:presso|sito\s+in)[\s:]+([^\n]{5,120})',
             ],
             validator_type="text",
             description="Luogo di esecuzione del contratto"),

    # ── Importi ────────────────────────────────────────────────────
    FieldDef("importo_totale", "Importo Totale", "💶", "Importi", "money",
             patterns=[
                 r"(?:importo\s+(?:complessivo|totale|massimo)\s+(?:dell['’]?\s*appalto)?)[^\n\u20ac]{0,40}\u20ac?\s*([0-9]{1,3}(?:[.,][0-9]{3})*[.,]\d{2})",
                 r'(?:valore\s+(?:stimato|complessivo|massimo))[^\n€]{0,40}€?\s*([0-9]{1,3}(?:[.,][0-9]{3})*[.,]\d{2})',
             ],
             validator_type="importi"),
    FieldDef("importo_base_gara", "Base di Gara", "💰", "Importi", "money",
             patterns=[
                 r"(?:importo\s+a\s+base\s+(?:di\s+)?(?:gara|d['’]?asta)|base\s+d['’]?asta)[^\n\u20ac]{0,40}\u20ac?\s*([0-9]{1,3}(?:[.,][0-9]{3})*[.,]\d{2})",
             ],
             validator_type="importi"),
    FieldDef("oneri_sicurezza", "Oneri Sicurezza", "🛡️", "Importi", "money",
             patterns=[
                 r'(?:oneri\s+(?:per\s+la\s+)?sicurezza|costi\s+(?:della\s+)?sicurezza)[^\n€]{0,50}€?\s*([0-9]{1,3}(?:[.,][0-9]{3})*[.,]\d{2})',
                 r'(?:oneri\s+(?:per\s+)?(?:interferenze|DUVRI))[^\n€]{0,50}€?\s*([0-9]{1,3}(?:[.,][0-9]{3})*[.,]\d{2})',
             ],
             validator_type="importi",
             description="Oneri per la sicurezza non soggetti a ribasso"),
    FieldDef("costi_manodopera", "Costi Manodopera", "👷", "Importi", "money",
             patterns=[
                 r'(?:costi?\s+(?:della?\s+)?manodopera)[^\n€]{0,50}€?\s*([0-9]{1,3}(?:[.,][0-9]{3})*[.,]\d{2})',
             ],
             validator_type="importi",
             description="Costi della manodopera ex art. 23 c.16"),
    FieldDef("garanzia_provvisoria", "Garanzia Provvisoria", "🔒", "Importi", "money",
             patterns=[
                 r'(?:garanzia\s+provvisoria|cauzione\s+provvisoria)[^\n€]{0,60}€?\s*([0-9]{1,3}(?:[.,][0-9]{3})*[.,]\d{2})',
                 r'(?:garanzia\s+provvisoria|cauzione\s+provvisoria)[^\n]{0,40}(\d[\d.,]+)\s*(?:€|euro)',
             ],
             validator_type="importi"),
    FieldDef("garanzia_definitiva", "Garanzia Definitiva", "🔒", "Importi", "text",
             patterns=[r'(?:garanzia\s+definitiva|cauzione\s+definitiva)[^\n]{0,80}?(\d+\s*%)'],
             validator_type="text"),
    FieldDef("contributo_anac", "Contributo ANAC", "🏦", "Importi", "money",
             patterns=[
                 r'(?:contributo\s+(?:ANAC|AVCP|Autorità))[^\n€]{0,40}€?\s*([0-9]{1,3}(?:[.,][0-9]{3})*[.,]\d{2})',
                 r'(?:contributo\s+gara)[^\n€]{0,40}€?\s*([0-9]{1,3}(?:[.,][0-9]{3})*[.,]\d{2})',
             ],
             validator_type="importi"),
    FieldDef("imposta_bollo", "Imposta di Bollo", "📄", "Importi", "money",
             patterns=[r'(?:imposta\s+di\s+bollo|marca\s+da\s+bollo)[^\n€]{0,40}€?\s*([0-9]{1,3}(?:[.,][0-9]{3})*[.,]\d{2})'],
             validator_type="importi"),
    FieldDef("anticipazione", "Anticipazione", "💸", "Importi", "text",
             patterns=[
                 r'(?:anticipazione)[^\n]{0,60}?(\d+\s*%)',
                 r'(?:anticipazione\s+del\s+prezzo)[^\n]{0,60}?(\d+\s*%)',
             ],
             validator_type="text",
             description="Percentuale di anticipazione del prezzo contrattuale"),

    # ── Punteggi OEPV ─────────────────────────────────────────────
    FieldDef("punteggio_tecnica", "Punti Tecnica", "⭐", "Punteggi OEPV", "number",
             patterns=[
                 r'(?:offerta|proposta|valutazione)\s+tecnica[^\n]{0,40}?(\d{1,3})\s*(?:punti|pt|/)',
                 r'(?:punteggio\s+tecnico|componente\s+tecnica)[^\n]{0,30}?(\d{1,3})',
             ],
             validator_type="punteggi"),
    FieldDef("punteggio_economica", "Punti Economica", "💰", "Punteggi OEPV", "number",
             patterns=[
                 r'(?:offerta|proposta|valutazione)\s+economic[ao][^\n]{0,40}?(\d{1,3})\s*(?:punti|pt|/)',
                 r'(?:punteggio\s+economic[ao]|componente\s+economic[ao])[^\n]{0,30}?(\d{1,3})',
             ],
             validator_type="punteggi"),
    FieldDef("soglia_sbarramento_tecnica", "Soglia Sbarramento", "⛔", "Punteggi OEPV", "number",
             patterns=[r'(?:soglia\s+(?:di\s+)?sbarramento|punteggio\s+minimo\s+tecnic[ao])[^\n]{0,30}?(\d{1,3})'],
             validator_type="punteggi"),
    FieldDef("pti_giovani_professionisti", "Bonus Giovani Prof.", "👨‍🎓", "Punteggi OEPV", "number",
             patterns=[r'giovan[ie]\s+professionisti?[^\n]{0,60}?(\d{1,2})\s*(?:punti|pt)'],
             validator_type="punteggi"),
    FieldDef("pti_iso_9001", "Bonus ISO 9001", "📋", "Punteggi OEPV", "number",
             patterns=[r'ISO\s*9001[^\n]{0,60}?(\d{1,2})\s*(?:punti|pt)'],
             validator_type="punteggi"),
    FieldDef("pti_parita_genere", "Bonus Parità Genere", "⚖️", "Punteggi OEPV", "number",
             patterns=[r'(?:parit[àa]\s+di\s+genere|certificazione\s+parit[àa])[^\n]{0,60}?(\d{1,2})\s*(?:punti|pt)'],
             validator_type="punteggi"),

    # ── Scadenze e Durata ──────────────────────────────────────────
    FieldDef("scadenza_offerte", "Scadenza Offerte", "📅", "Scadenze e Durata", "date",
             patterns=[
                 r'(?:scadenza|termine)[^\n]{0,20}(?:presentazione|ricezione)[^\n]{0,20}offert[ae][^\n]{0,30}(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}(?:\s+(?:ore|h)\s*\d{1,2}[:.]\d{2})?)',
                 r'(?:entro\s+(?:e\s+non\s+oltre\s+)?(?:le\s+ore\s+)?\d{1,2}[:.]\d{2}\s+del\s+)(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})',
             ],
             validator_type="date"),
    FieldDef("termine_chiarimenti", "Termine Chiarimenti", "❓", "Scadenze e Durata", "text",
             patterns=[r'(?:termine|scadenza)[^\n]{0,20}(?:chiarimenti|quesiti)[^\n]{0,30}(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}[^\n]{0,30})'],
             validator_type="text"),
    FieldDef("durata_contratto", "Durata Contratto", "⏱", "Scadenze e Durata", "text",
             patterns=[
                 r'(?:durata\s+(?:del\s+)?(?:contratto|servizio|incarico|appalto))[^\n]{0,60}?(\d+\s*(?:giorn[io]|mes[ie]|ann[io]|settiman[ae])[^\n]{0,40})',
                 r'(?:tempi?\s+(?:di\s+)?esecuzione|termine\s+(?:di\s+)?(?:esecuzione|ultimazione))[^\n]{0,60}?(\d+\s*(?:giorn[io]|mes[ie]|ann[io]|settiman[ae])[^\n]{0,40})',
             ],
             validator_type="text"),
    FieldDef("termine_esecuzione", "Termine Esecuzione", "⏳", "Scadenze e Durata", "text",
             patterns=[
                 r'(?:termine\s+(?:di\s+)?(?:esecuzione|ultimazione|completamento))[^\n]{0,60}?(\d+\s*(?:giorn[io]|mes[ie]|ann[io]))',
                 r"(?:tempo\s+(?:utile|massimo)\s+(?:per\s+)?(?:l['’]?\s*)?esecuzione)[^\n]{0,60}?(\d+\s*(?:giorn[io]|mes[ie]|ann[io]))",
             ],
             validator_type="text",
             description="Termine massimo per l'esecuzione dei lavori/servizi"),
    FieldDef("piattaforma_url", "Piattaforma", "🌐", "Scadenze e Durata", "text", full_width=True,
             patterns=[r'((?:https?://)?(?:www\.)?[a-zA-Z0-9\-]+\.(?:it|eu|com|gov\.it)/[^\s\)]{3,80})'],
             validator_type="text"),

    # ── Requisiti e Categorie ──────────────────────────────────────
    FieldDef("fatturato_minimo", "Fatturato Minimo", "📈", "Requisiti e Categorie", "money",
             patterns=[r'(?:fatturato\s+(?:globale|specifico|minimo))[^\n€]{0,60}€?\s*([0-9]{1,3}(?:[.,][0-9]{3})*[.,]\d{2})'],
             validator_type="importi"),
    FieldDef("periodo_requisiti_anni", "Periodo Requisiti", "📅", "Requisiti e Categorie", "number",
             patterns=[r'(?:ultim[io]\s+|negli?\s+ultim[io]\s+)(\d+)\s*(?:anni|eserciz[io])'],
             validator_type="punteggi"),
    FieldDef("categorie_ingegneria", "Categorie Ingegneria", "🏗", "Requisiti e Categorie", "list",
             description="Categorie DM 17/06/2016 (E, S, IA, ecc.)"),
    FieldDef("polizza_professionale", "Polizza Professionale", "🛡️", "Requisiti e Categorie", "text",
             patterns=[
                 r'(?:polizza\s+(?:assicurativa\s+)?professionale|RC\s+professionale)[^\n€]{0,60}€?\s*([0-9]{1,3}(?:[.,][0-9]{3})*[.,]\d{2})',
                 r'(?:polizza\s+(?:assicurativa\s+)?professionale|copertura\s+assicurativa)[^\n]{0,120}',
             ],
             validator_type="text",
             description="Polizza RC professionale richiesta"),
    FieldDef("requisiti_soa", "Requisiti SOA", "🏗️", "Requisiti e Categorie", "text",
             patterns=[
                 r'(?:categori[ae]\s+SOA|qualificazione\s+SOA|class[ei]\s+SOA)[^\n]{0,120}',
                 r'(?:OG|OS)\s*\d{1,2}\s*[–\-]\s*class(?:ific)?[ae]?\s*(?:I{1,3}V?|V|VI{0,3})',
             ],
             validator_type="text",
             description="Categorie e classifiche SOA richieste"),

    # ── Condizioni e Clausole ──────────────────────────────────────
    FieldDef("sopralluogo_obbligatorio", "Sopralluogo Obbligatorio", "🚶", "Condizioni e Clausole", "boolean",
             description="Indica se il sopralluogo è obbligatorio"),
    FieldDef("sopralluogo_note", "Note Sopralluogo", "🚶", "Condizioni e Clausole", "text", full_width=True,
             patterns=[r'sopralluogo[^\n]{0,300}'],
             validator_type="text",
             description="Dettagli/modalità del sopralluogo"),
    FieldDef("subappalto", "Subappalto", "🔗", "Condizioni e Clausole", "text", full_width=True,
             patterns=[
                 r'(?:subappalto|sub[\s-]?appalto)[^\n]{0,250}',
                 r'(?:subappalto)[^\n]{0,40}?(\d+\s*%)',
             ],
             validator_type="text"),
    FieldDef("avvalimento", "Avvalimento", "🤝", "Condizioni e Clausole", "text", full_width=True,
             patterns=[r'(?:avvalimento)[^\n]{0,250}'],
             validator_type="text"),
    FieldDef("verifica_anomalia", "Verifica Anomalia", "📉", "Condizioni e Clausole", "text", full_width=True,
             patterns=[r'(?:verifica\s+(?:della?\s+)?(?:congruità|anomalia)|offert[ae]\s+anomal[ae])[^\n]{0,200}'],
             validator_type="text"),
    FieldDef("finanziamento", "Fonte Finanziamento", "🇪🇺", "Condizioni e Clausole", "text",
             description="Classificato dal motore AI (PNRR, FESR, etc.)"),
    FieldDef("revisione_prezzi", "Revisione Prezzi", "📊", "Condizioni e Clausole", "boolean"),
    FieldDef("conformita_cam", "Conformità CAM", "🌿", "Condizioni e Clausole", "boolean"),
    FieldDef("inversione_procedimentale", "Inversione Proc.", "🔄", "Condizioni e Clausole", "boolean"),
    FieldDef("soccorso_istruttorio", "Soccorso Istruttorio", "📨", "Condizioni e Clausole", "boolean",
             description="Previsto soccorso istruttorio per documentazione incompleta"),
    FieldDef("clausola_sociale", "Clausola Sociale", "👥", "Condizioni e Clausole", "boolean",
             description="Obbligo di assorbimento personale uscente"),
    FieldDef("penali", "Penali", "⚠️", "Condizioni e Clausole", "text", full_width=True,
             patterns=[
                 r'(?:penali?\s+(?:per\s+)?(?:ritard[io]|inadempimento))[^\n]{0,200}',
                 r'(?:penale)[^\n]{0,40}?(\d+[.,]?\d*\s*%[^\n]{0,60})',
             ],
             validator_type="text",
             description="Penali contrattuali"),
    FieldDef("modalita_pagamento", "Modalità Pagamento", "💳", "Condizioni e Clausole", "text", full_width=True,
             patterns=[
                 r'(?:modalit[àa]\s+(?:di\s+)?pagamento|termini?\s+(?:di\s+)?pagamento)[^\n]{0,200}',
                 r'(?:pagamento)[^\n]{0,40}?(\d+\s*giorni[^\n]{0,60})',
             ],
             validator_type="text",
             description="Terminidi e modalità di pagamento"),
    FieldDef("garanzia_provvisoria_ridotta", "Riduzione Garanzia", "🔓", "Condizioni e Clausole", "text",
             patterns=[
                 r'(?:riduz(?:ione|\.)\s+(?:della?\s+)?(?:garanzia|cauzione))[^\n]{0,150}',
                 r'(?:ridotta\s+del|riduzione\s+del?)\s+(\d+\s*%)',
             ],
             validator_type="text",
             description="Riduzioni garanzia provvisoria ammesse"),
]


# ═══════════════════════════════════════════════════════════════════════════
# REGISTRY — gestisce campi built-in + custom (DB)
# ═══════════════════════════════════════════════════════════════════════════

class FieldRegistry:
    """Registro centralizzato dei campi. Unisce built-in + custom da DB."""

    def __init__(self):
        self._builtin: Dict[str, FieldDef] = {f.key: f for f in BUILTIN_FIELDS}
        self._custom: Dict[str, FieldDef] = {}
        init_custom_fields_table()
        self._load_custom_fields()

    def _load_custom_fields(self):
        """Carica campi custom dal database."""
        try:
            with get_connection(readonly=True) as conn:
                rows = conn.execute("SELECT * FROM custom_fields").fetchall()
            for r in rows:
                fd = FieldDef(
                    key=r[0], label=r[1], icon=r[2] or "📋",
                    category=r[3] or "Custom",
                    field_type=r[4] or "text",
                    mono=bool(r[5]), highlight=bool(r[6]), full_width=bool(r[7]),
                    patterns=json.loads(r[8] or "[]"),
                    validator_type=r[9] or "text",
                    is_custom=True,
                    description=r[10] or "",
                    extraction_hint=r[11] or "",
                )
                self._custom[fd.key] = fd
        except Exception:
            pass

    # ── Accesso campi ──────────────────────────────────────────────

    def get_all(self) -> List[FieldDef]:
        """Tutti i campi (built-in + custom)."""
        merged = dict(self._builtin)
        merged.update(self._custom)
        return list(merged.values())

    def get(self, key: str) -> Optional[FieldDef]:
        """Restituisce un campo per chiave."""
        return self._custom.get(key) or self._builtin.get(key)

    def get_by_category(self) -> Dict[str, List[FieldDef]]:
        """Raggruppa tutti i campi per categoria (per UI)."""
        cats: Dict[str, List[FieldDef]] = {}
        for f in self.get_all():
            cats.setdefault(f.category, []).append(f)
        return cats

    def get_keys(self) -> List[str]:
        """Tutte le chiavi campo."""
        return [f.key for f in self.get_all()]

    def get_patterns(self) -> Dict[str, List[str]]:
        """Dizionario key → [patterns] per estrazione."""
        return {f.key: f.patterns for f in self.get_all() if f.patterns}

    def get_validators(self) -> Dict[str, Callable]:
        """Dizionario key → validator function."""
        result = {}
        for f in self.get_all():
            if f.validator_type:
                v = get_validator(f.validator_type)
                if v:
                    result[f.key] = v
        return result

    # ── Gestione campi custom ──────────────────────────────────────

    def add_custom_field(self, data: dict) -> FieldDef:
        """Aggiunge un campo custom. data deve avere almeno 'key' e 'label'."""
        key = data.get("key", "").strip().lower().replace(" ", "_")
        label = data.get("label", "").strip()
        if not key or not label:
            raise ValueError("Chiave e label obbligatorie")
        if key in self._builtin:
            raise ValueError(f"La chiave '{key}' è riservata a un campo built-in")

        fd = FieldDef(
            key=key,
            label=label,
            icon=data.get("icon", "📋"),
            category=data.get("category", "Custom"),
            field_type=data.get("field_type", "text"),
            mono=data.get("mono", False),
            highlight=data.get("highlight", False),
            full_width=data.get("full_width", False),
            patterns=data.get("patterns", []),
            validator_type=data.get("validator_type", "text"),
            is_custom=True,
            description=data.get("description", ""),
            extraction_hint=data.get("extraction_hint", ""),
        )

        now = datetime.now().isoformat()
        with get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO custom_fields
                (key, label, icon, category, field_type, mono, highlight, full_width,
                 patterns, validator_type, description, extraction_hint, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                fd.key, fd.label, fd.icon, fd.category, fd.field_type,
                int(fd.mono), int(fd.highlight), int(fd.full_width),
                json.dumps(fd.patterns), fd.validator_type,
                fd.description, fd.extraction_hint, now, now,
            ))
        self._custom[fd.key] = fd
        return fd

    def update_custom_field(self, key: str, data: dict) -> FieldDef:
        """Aggiorna un campo custom esistente."""
        if key not in self._custom:
            raise ValueError(f"Campo custom '{key}' non trovato")
        fd = self._custom[key]
        # Aggiorna campi forniti
        for attr in ("label", "icon", "category", "field_type", "mono", "highlight",
                      "full_width", "validator_type", "description", "extraction_hint"):
            if attr in data:
                setattr(fd, attr, data[attr])
        if "patterns" in data:
            fd.patterns = data["patterns"]

        now = datetime.now().isoformat()
        with get_connection() as conn:
            conn.execute("""
                UPDATE custom_fields SET label=?, icon=?, category=?, field_type=?,
                mono=?, highlight=?, full_width=?, patterns=?, validator_type=?,
                description=?, extraction_hint=?, updated_at=?
                WHERE key=?
            """, (
                fd.label, fd.icon, fd.category, fd.field_type,
                int(fd.mono), int(fd.highlight), int(fd.full_width),
                json.dumps(fd.patterns), fd.validator_type,
                fd.description, fd.extraction_hint, now, key,
            ))
        return fd

    def delete_custom_field(self, key: str) -> bool:
        """Elimina un campo custom."""
        if key not in self._custom:
            return False
        with get_connection() as conn:
            conn.execute("DELETE FROM custom_fields WHERE key=?", (key,))
        del self._custom[key]
        return True

    def get_custom_fields(self) -> List[FieldDef]:
        """Solo campi custom."""
        return list(self._custom.values())

    # ── Export per frontend ────────────────────────────────────────

    def to_sections_json(self) -> List[dict]:
        """Genera la struttura SECTIONS per il frontend (come nell'HTML attuale)."""
        CATEGORY_ICONS = {
            "Identificativi": "🔑", "Soggetti": "🏛",
            "Oggetto e Procedura": "📋", "Importi": "💶",
            "Punteggi OEPV": "⭐", "Scadenze e Durata": "📅",
            "Requisiti e Categorie": "📊", "Condizioni e Clausole": "📝",
            "Custom": "🔧",
        }
        # Ordine categorie
        CAT_ORDER = [
            "Identificativi", "Soggetti", "Oggetto e Procedura",
            "Importi", "Punteggi OEPV", "Scadenze e Durata",
            "Requisiti e Categorie", "Condizioni e Clausole", "Custom",
        ]
        cats = self.get_by_category()
        sections = []
        # Prima le categorie nell'ordine definito
        for cat in CAT_ORDER:
            if cat in cats:
                fields = cats.pop(cat)
                sections.append({
                    "title": cat,
                    "icon": CATEGORY_ICONS.get(cat, "📋"),
                    "fields": [f.to_dict() for f in fields],
                })
        # Poi le categorie restanti (custom)
        for cat, fields in cats.items():
            sections.append({
                "title": cat,
                "icon": CATEGORY_ICONS.get(cat, "📋"),
                "fields": [f.to_dict() for f in fields],
            })
        return sections


# ── Singleton globale ──────────────────────────────────────────────────
registry = FieldRegistry()
