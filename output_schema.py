"""
AppaltoAI — Schema di Output Strutturato
Pydantic models che definiscono la struttura JSON target dell'estrazione.
Rispecchia esattamente la struttura di info.json (il "prompt").
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ═════════════════════════════════════════════════════════════════════════════
# SUB-MODELLI
# ═════════════════════════════════════════════════════════════════════════════

class PrestazioneLotto(BaseModel):
    """Singola prestazione di un lotto."""
    descrizione: Optional[str] = None
    codice_CPV: Optional[str] = None
    tipo: Optional[str] = None
    importo: Optional[float] = None


class CategoriaServizi(BaseModel):
    """Categoria di servizi con importo."""
    codice: Optional[str] = None
    descrizione: Optional[str] = None
    importo_complessivo_lavori_progettati_euro: Optional[float] = None
    grado_complessita: Optional[float] = None
    lotto: Optional[int] = None  # riferimento al lotto specifico, se applicabile


class LottoImporto(BaseModel):
    """Singolo lotto con importi e quote."""
    lotto: int
    ubicazione: Optional[str] = None
    importo_euro: Optional[float] = None
    quota_fissa_65_percento_euro: Optional[float] = None
    quota_ribassabile_35_percento_euro: Optional[float] = None
    prestazioni: List[PrestazioneLotto] = Field(default_factory=list)
    categorie: List[CategoriaServizi] = Field(default_factory=list)


class DescrizioneLavori(BaseModel):
    """Descrizione lavori con suddivisione per lotti e importo totale."""
    tipologia: Optional[str] = None
    lotti: List[LottoImporto] = Field(default_factory=list)
    importo_totale_procedura_euro: Optional[float] = None
    note: Optional[str] = None


class RUP(BaseModel):
    """Responsabile Unico del Procedimento."""
    nome: Optional[str] = None
    qualifica: Optional[str] = None  # Dott., Dott.ssa, Ing., Arch., ecc.
    ruolo: Optional[str] = None  # es. "Programmazione/Progettazione/Esecuzione" o "Affidamento (CUC)"
    email: Optional[str] = None


class StazioneAppaltante(BaseModel):
    """Stazione appaltante / ente."""
    ente: Optional[str] = None
    ente_delegante: Optional[str] = None  # ente che ha delegato la procedura (es. Comune di X)
    cuc: Optional[str] = None  # Centrale Unica di Committenza (stazione appaltante delegata)
    rup: Optional[RUP] = None  # RUP principale (fasi programmazione/progettazione/esecuzione)
    rup_cuc: Optional[RUP] = None  # RUP designato dalla CUC per la fase di affidamento
    sede: Optional[str] = None


class ProfiloRichiesto(BaseModel):
    """Profilo professionale richiesto."""
    numero: int = 1
    ruolo: Optional[str] = None
    requisiti: Optional[str] = None
    laurea: Optional[str] = None
    diploma: Optional[str] = None
    abilitazione: Optional[str] = None
    iscrizione_albo: Optional[str] = None
    anni_esperienza: Optional[int] = None
    esperienza_servizi: Optional[str] = None
    certificazione: Optional[str] = None
    riferimento_normativo: Optional[str] = None


class RequisitiIdoneitaProfessionale(BaseModel):
    """Requisiti di idoneità professionale."""
    profili_richiesti: List[ProfiloRichiesto] = Field(default_factory=list)
    numero_minimo_professionisti: Optional[int] = None
    ruoli_cumulabili: Optional[bool] = None
    iscrizioni_richieste: List[str] = Field(default_factory=list)
    note: Optional[str] = None


# CategoriaServizi is defined earlier (before LottoImporto) so it can be used by both LottoImporto and ServiziDiPunta


class ServiziDiPunta(BaseModel):
    """Servizi di punta richiesti."""
    numero: Optional[int] = None
    periodo: Optional[str] = None
    tipologia: Optional[str] = None
    categorie: List[CategoriaServizi] = Field(default_factory=list)
    note: Optional[str] = None


class RequisitiMezzi(BaseModel):
    """Requisiti sui mezzi/veicoli richiesti."""
    numero_minimo: Optional[int] = None
    attrezzature_richieste: List[str] = Field(default_factory=list)
    note: Optional[str] = None


class RequisitiCapacitaTecnica(BaseModel):
    """Requisiti di capacità tecnico-professionale."""
    servizi_di_punta: Optional[ServiziDiPunta] = None
    importo_minimo_servizi_analoghi_euro: Optional[float] = None
    periodo_servizi_analoghi: Optional[str] = None
    ccnl: Optional[str] = None
    requisiti_mezzi: Optional[RequisitiMezzi] = None


class RequisitiCapacitaEconomica(BaseModel):
    """Requisiti di capacità economico-finanziaria."""
    descrizione: Optional[str] = None
    importo_minimo_euro: Optional[float] = None
    note: Optional[str] = None


class Sopralluogo(BaseModel):
    """Info sopralluogo."""
    obbligatorio: bool = False
    note: Optional[str] = None


class GestorePiattaforma(BaseModel):
    """Contatti gestore piattaforma."""
    telefono_assistenza: Optional[str] = None
    email_assistenza: Optional[str] = None


class RegolePresentazioneOfferte(BaseModel):
    """Regole per la presentazione delle offerte."""
    piattaforma: Optional[str] = None
    modalita: Optional[str] = None
    gestore: Optional[GestorePiattaforma] = None


class SubCriterio(BaseModel):
    """Sub-criterio di valutazione."""
    codice: Optional[str] = None
    punteggio: Optional[float] = None
    punteggio_discrezionale: Optional[float] = None
    punteggio_tabellare: Optional[float] = None
    tipo: Optional[str] = None
    descrizione: Optional[str] = None
    descrizione_dettagliata: Optional[str] = None


class CriterioValutazione(BaseModel):
    """Criterio di valutazione dell'offerta tecnica."""
    codice: Optional[str] = None
    nome: Optional[str] = None
    punteggio: Optional[float] = None
    punteggio_discrezionale: Optional[float] = None
    punteggio_tabellare: Optional[float] = None
    tipo: Optional[str] = None
    descrizione: Optional[str] = None
    descrizione_dettagliata: Optional[str] = None
    sub_criteri: List[SubCriterio] = Field(default_factory=list)


class OffertaTecnica(BaseModel):
    """Struttura offerta tecnica con criteri."""
    punteggio_massimo: Optional[float] = None
    formato_relazione: Optional[str] = None
    contenuto_busta_tecnica: List[str] = Field(default_factory=list)
    note_importanti: List[str] = Field(default_factory=list)
    criteri: List[CriterioValutazione] = Field(default_factory=list)


class OffertaEconomica(BaseModel):
    """Struttura offerta economica."""
    punteggio_massimo: Optional[float] = None
    modalita: Optional[str] = None
    note: Optional[str] = None


class RipartizioneValutazione(BaseModel):
    """Ripartizione punteggi offerta."""
    offerta_tecnica: Optional[float] = None
    offerta_economica: Optional[float] = None


class MetodoValutazione(BaseModel):
    """Metodo di valutazione."""
    tecnica: Optional[str] = None
    economica: Optional[str] = None


class CriteriValutazione(BaseModel):
    """Schema complessivo dei criteri di valutazione."""
    punteggio_totale: Optional[float] = None
    ripartizione: Optional[RipartizioneValutazione] = None
    metodo_valutazione: Optional[MetodoValutazione] = None


class VincoliPartecipazione(BaseModel):
    """Vincoli di partecipazione alla gara."""
    vincolo_partecipazione_entrambi_lotti: bool = False
    offerta_identica_per_entrambi_lotti: bool = False
    medesima_forma_giuridica: bool = False
    vincolo_aggiudicazione: Optional[str] = None
    nota_difformita: Optional[str] = None


class TempistichEsecuzione(BaseModel):
    """Tempistiche di esecuzione del contratto."""
    rilievi_geometrici_architettonici_giorni: Optional[int] = None
    progetto_esecutivo_giorni: Optional[int] = None
    relazioni_rettifiche_integrazioni_giorni: Optional[int] = None
    durata_complessiva_stimata_giorni: Optional[int] = None
    note: Optional[str] = None


class GaranziaProvvisoria(BaseModel):
    """Garanzia provvisoria."""
    richiesta: bool = False
    percentuale: Optional[float] = None
    importo: Optional[float] = None
    note: Optional[str] = None


class GaranziaDefinitiva(BaseModel):
    """Garanzia definitiva."""
    richiesta: bool = False
    percentuale: Optional[float] = None
    note: Optional[str] = None


class PolizzaRC(BaseModel):
    """Polizza RC professionale."""
    richiesta: bool = False
    copertura: Optional[str] = None
    note: Optional[str] = None


class RevisionePrezzi(BaseModel):
    """Revisione prezzi contrattuale."""
    ammessa: bool = False
    soglia_percentuale: Optional[float] = None
    percentuale_applicabile: Optional[float] = None
    indice_riferimento: Optional[str] = None
    note: Optional[str] = None


# ═════════════════════════════════════════════════════════════════════════════
# MODELLO PRINCIPALE — Output Appalto
# ═════════════════════════════════════════════════════════════════════════════

class AppaltoOutput(BaseModel):
    """Schema di output completo per l'estrazione da disciplinare di gara.
    Corrisponde alla struttura target definita in info.json."""

    # Identificativi
    cig: Optional[Dict[str, str]] = None
    cup: Optional[str] = None
    codice_NUTS: Optional[str] = None
    CPV_principale: Optional[str] = None
    procedura_tipo: Optional[str] = None
    criterio_aggiudicazione: Optional[str] = None

    # Oggetto
    oggetto_appalto: Optional[str] = None

    # Lavori e importi
    descrizione_lavori_con_importo_totale: Optional[DescrizioneLavori] = None

    # Soggetti
    stazione_appaltante: Optional[StazioneAppaltante] = None

    # Procedura
    tipologia_appalto: Optional[str] = None

    # Requisiti
    requisiti_idoneita_professionale: Optional[RequisitiIdoneitaProfessionale] = None
    requisiti_capacita_tecnica_professionale: Optional[RequisitiCapacitaTecnica] = None
    requisiti_capacita_economica_finanziaria: Optional[RequisitiCapacitaEconomica] = None

    # Sopralluogo
    sopralluogo: Optional[Sopralluogo] = None

    # Scadenza
    scadenza: Optional[str] = None

    # Offerte
    regole_presentazione_offerte: Optional[RegolePresentazioneOfferte] = None
    documentazione_amministrativa_richiesta: List[str] = Field(default_factory=list)

    # Criteri e punteggi
    offerta_tecnica: Optional[OffertaTecnica] = None
    offerta_economica: Optional[OffertaEconomica] = None
    criteri_valutazione_offerta_tecnica: Optional[CriteriValutazione] = None

    # Vincoli
    vincoli_partecipazione: Optional[VincoliPartecipazione] = None

    # Tempistiche
    tempistiche_esecuzione: Optional[TempistichEsecuzione] = None

    # Garanzie
    garanzia_provvisoria: Optional[GaranziaProvvisoria] = None
    garanzia_definitiva: Optional[GaranziaDefinitiva] = None
    polizza_RC_professionale: Optional[PolizzaRC] = None

    # CAM
    CAM_criteri_ambientali: Optional[str] = None

    # Revisione prezzi
    revisione_prezzi: Optional[RevisionePrezzi] = None

    # Note
    note_particolari: List[str] = Field(default_factory=list)
