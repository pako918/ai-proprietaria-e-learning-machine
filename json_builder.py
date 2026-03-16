"""
AppaltoAI — JSON Builder
═════════════════════════
Trasforma il dizionario nested prodotto da extract_rules_based()
nella struttura target definita in output_schema.py (info.json).

Sostituisce flatten_for_pipeline() come unico punto di uscita dati.
"""

from typing import Optional

from output_schema import (
    AppaltoOutput,
    LottoImporto,
    PrestazioneLotto,
    DescrizioneLavori,
    RUP,
    StazioneAppaltante,
    ProfiloRichiesto,
    RequisitiIdoneitaProfessionale,
    CategoriaServizi,
    ServiziDiPunta,
    RequisitiCapacitaTecnica,
    RequisitiCapacitaEconomica,
    RequisitiMezzi,
    Sopralluogo,
    GestorePiattaforma,
    RegolePresentazioneOfferte,
    SubCriterio,
    CriterioValutazione,
    OffertaTecnica,
    OffertaEconomica,
    RipartizioneValutazione,
    MetodoValutazione,
    CriteriValutazione,
    VincoliPartecipazione,
    TempistichEsecuzione,
    GaranziaProvvisoria,
    GaranziaDefinitiva,
    PolizzaRC,
    RevisionePrezzi,
)


def _g(d: dict, *keys, default=None):
    """Navigazione sicura di un dict annidato."""
    obj = d
    for k in keys:
        if isinstance(obj, dict):
            obj = obj.get(k)
        else:
            return default
    return obj if obj is not None else default


def _parse_float(val) -> Optional[float]:
    """Converte un valore numerico in float, se possibile."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def build_output(nested: dict) -> dict:
    """
    Costruisce la struttura AppaltoOutput a partire dal risultato
    nested di extract_rules_based().

    Returns:
        dict serializzabile JSON con la struttura target (info.json).
    """
    ig = nested.get("informazioni_generali", {})
    tp = nested.get("tipo_procedura", {})
    pt = nested.get("piattaforma_telematica", {})
    sl = nested.get("suddivisione_lotti", {})
    ic = nested.get("importi_complessivi", {})
    temp = nested.get("tempistiche", {})
    rp = nested.get("requisiti_partecipazione", {})
    cv = nested.get("criteri_valutazione", {})
    gar = nested.get("garanzie", {})
    sop = nested.get("sopralluogo", {})
    ia = nested.get("informazioni_aggiuntive", {})
    doc = nested.get("documentazione_amministrativa", {})
    agg = nested.get("aggiudicazione", {})
    otf = nested.get("offerta_tecnica_formato", {})

    # ── CIG ──
    cig_dict = None
    cig_val = ig.get("CIG")
    cig_per_lotto = ig.get("CIG_per_lotto", [])
    if cig_per_lotto:
        cig_dict = {f"lotto_{c['lotto']}": c["CIG"] for c in cig_per_lotto if isinstance(c, dict)}
    elif cig_val:
        cig_dict = {"lotto_1": cig_val}

    # ── CUP ──
    cup = ig.get("CUP")

    # ── Oggetto ──
    oggetto = ig.get("titolo")

    # ── Descrizione lavori con importo totale ──
    lotti_raw = sl.get("lotti", [])
    lotti_output = []
    for lotto in lotti_raw:
        if not isinstance(lotto, dict):
            continue
        importo = _parse_float(lotto.get("importo_base_asta"))
        quota_fissa = None
        quota_rib = None
        costo_mano = _parse_float(lotto.get("costo_manodopera") or ic.get("costo_manodopera"))
        imp_rib_dir = _parse_float(lotto.get("importo_soggetto_ribasso") or ic.get("importo_totale_soggetto_ribasso"))
        perc_rib = lotto.get("quota_ribassabile_percentuale") or ic.get("quota_ribassabile_percentuale")
        if costo_mano and importo:
            quota_fissa = costo_mano
            quota_rib = round(importo - costo_mano, 2)
        elif imp_rib_dir and importo:
            quota_rib = imp_rib_dir
            quota_fissa = round(importo - imp_rib_dir, 2)
        elif importo and perc_rib:
            perc_fissa = 100 - perc_rib
            quota_fissa = round(importo * perc_fissa / 100, 2)
            quota_rib = round(importo * perc_rib / 100, 2)

        lotti_output.append(LottoImporto(
            lotto=lotto.get("numero", 1),
            ubicazione=lotto.get("luogo_esecuzione") or lotto.get("ubicazione"),  # noqa: E501
            importo_euro=importo,
            quota_fissa_65_percento_euro=quota_fissa,
            quota_ribassabile_35_percento_euro=quota_rib,
            prestazioni=[
                PrestazioneLotto(
                    descrizione=p.get("descrizione"),
                    codice_CPV=p.get("codice_CPV"),
                    tipo=p.get("tipo"),
                    importo=_parse_float(p.get("importo")),
                )
                for p in lotto.get("prestazioni", [])
                if isinstance(p, dict)
            ],
        ))

    # Tipologia: determiniamo dal tipo_procedura
    tipologia_lavori = None
    criterio = tp.get("criterio_aggiudicazione", "")
    tipo_proc = tp.get("tipo", "")
    if tipo_proc or criterio:
        parts = []
        if tipo_proc:
            parts.append(f"Procedura {tipo_proc}")
        rif = tp.get("riferimento_normativo")
        if rif:
            parts.append(f"({rif})")
        if criterio:
            if criterio == "OEPV":
                parts.append("con criterio offerta economicamente più vantaggiosa")
            elif criterio == "minor_prezzo":
                parts.append("con criterio del minor prezzo")
        tipologia_lavori = " ".join(parts) if parts else None

    importo_totale = _parse_float(ic.get("importo_totale_gara"))

    # Note lavori
    note_lavori_parts = []
    if ic.get("quota_ribassabile_percentuale"):
        perc = ic["quota_ribassabile_percentuale"]
        note_lavori_parts.append(
            f"Il {100 - perc}% è prezzo fisso non soggetto a ribasso, il {perc}% è ribassabile"
        )
    note_lavori = ". ".join(note_lavori_parts) if note_lavori_parts else None

    descr_lavori = DescrizioneLavori(
        tipologia=tipologia_lavori,
        lotti=lotti_output,
        importo_totale_procedura_euro=importo_totale,
        note=note_lavori,
    ) if (lotti_output or importo_totale) else None

    # ── Stazione appaltante ──
    sa_raw = ig.get("stazione_appaltante", {})
    rup_raw = ig.get("RUP", ig.get("rup", {}))
    rup_cuc_raw = ig.get("RUP_CUC", {})
    sa = None
    if sa_raw or rup_raw or rup_cuc_raw:
        def _build_rup_obj(r: dict):
            if not isinstance(r, dict) or not r.get("nome"):
                return None
            qualifica = r.get("qualifica", "")
            nome = r["nome"]
            nome_completo = f"{qualifica} {nome}".strip() if qualifica else nome
            return RUP(
                nome=nome_completo,
                qualifica=r.get("qualifica"),
                ruolo=r.get("ruolo"),
                email=r.get("email"),
            )

        rup_obj = _build_rup_obj(rup_raw)
        rup_cuc_obj = _build_rup_obj(rup_cuc_raw)

        sede_parts = []
        for k in ["indirizzo", "sede", "citta"]:
            v = sa_raw.get(k)
            if v:
                sede_parts.append(v)
        sede = ", ".join(sede_parts) if sede_parts else None

        sa = StazioneAppaltante(
            ente=sa_raw.get("denominazione"),
            ente_delegante=sa_raw.get("ente_delegante"),
            cuc=sa_raw.get("CUC"),
            rup=rup_obj,
            rup_cuc=rup_cuc_obj,
            sede=sede,
        )

    # ── Tipologia appalto ──
    tipologia_appalto = tipologia_lavori  # stessa stringa costruita prima

    # ── Requisiti idoneità professionale ──
    gdl = rp.get("gruppo_di_lavoro", {})
    figure = gdl.get("figure_professionali", [])
    profili = []
    for i, fig in enumerate(figure):
        if not isinstance(fig, dict):
            continue
        det = fig.get("dettaglio", {})
        profili.append(ProfiloRichiesto(
            numero=fig.get("numero", i + 1),
            ruolo=fig.get("ruolo"),
            requisiti=fig.get("requisiti"),
            laurea=det.get("laurea"),
            diploma=det.get("diploma"),
            abilitazione=det.get("abilitazione"),
            iscrizione_albo=det.get("iscrizione_albo"),
            anni_esperienza=det.get("anni_esperienza"),
            esperienza_servizi=det.get("esperienza_servizi"),
            certificazione=det.get("certificazione"),
            riferimento_normativo=det.get("riferimento_normativo"),
        ))

    req_idon = None
    if profili or rp.get("idoneita_professionale", {}).get("iscrizioni_richieste"):
        note_idon = None
        sogg = rp.get("soggetti_ammessi", {})
        if isinstance(sogg, dict) and sogg.get("obbligo_giovane_professionista"):
            note_idon = "Obbligo giovane professionista nel gruppo di lavoro"
        req_idon = RequisitiIdoneitaProfessionale(
            profili_richiesti=profili,
            numero_minimo_professionisti=gdl.get("numero_minimo_professionisti"),
            ruoli_cumulabili=gdl.get("ruoli_cumulabili"),
            iscrizioni_richieste=rp.get("idoneita_professionale", {}).get("iscrizioni_richieste", []),
            note=note_idon,
        )

    # ── Requisiti capacità tecnica ──
    ctp = rp.get("capacita_tecnico_professionale", {})
    srv = ctp.get("servizi_analoghi", {})
    req_tec = None
    _has_srv = isinstance(srv, dict) and srv.get("richiesti")
    _importo_min_srv = _parse_float(srv.get("importo_minimo")) if isinstance(srv, dict) else None
    _ccnl_val = ctp.get("ccnl") or (srv.get("ccnl") if isinstance(srv, dict) else None)
    _mezzi_raw = ctp.get("requisiti_mezzi") or (srv.get("requisiti_mezzi") if isinstance(srv, dict) else None)
    _req_mezzi_obj = None
    if isinstance(_mezzi_raw, dict):
        _req_mezzi_obj = RequisitiMezzi(
            numero_minimo=_mezzi_raw.get("numero_minimo"),
            attrezzature_richieste=_mezzi_raw.get("attrezzature_richieste", []),
            note=_mezzi_raw.get("note"),
        )
    if _has_srv or _importo_min_srv or _ccnl_val or _req_mezzi_obj:
        categorie_srv = []
        if _has_srv:
            for cat in srv.get("categorie_richieste", []):
                if not isinstance(cat, dict):
                    continue
                categorie_srv.append(CategoriaServizi(
                    codice=cat.get("categoria"),
                    descrizione=cat.get("descrizione"),
                    importo_complessivo_lavori_progettati_euro=_parse_float(cat.get("importo_minimo")),
                ))
        servizi_punta = None
        if _has_srv or categorie_srv:
            servizi_punta = ServiziDiPunta(
                numero=srv.get("numero_servizi") if isinstance(srv, dict) else None,
                periodo=srv.get("periodo_riferimento") if isinstance(srv, dict) else None,
                tipologia=srv.get("tipologia") if isinstance(srv, dict) else None,
                categorie=categorie_srv,
                note=srv.get("note") if isinstance(srv, dict) else None,
            )
        req_tec = RequisitiCapacitaTecnica(
            servizi_di_punta=servizi_punta,
            importo_minimo_servizi_analoghi_euro=_importo_min_srv,
            periodo_servizi_analoghi=srv.get("periodo_riferimento") if isinstance(srv, dict) else None,
            ccnl=_ccnl_val,
            requisiti_mezzi=_req_mezzi_obj,
        )

    # ── Requisiti capacità economica ──
    cef = rp.get("capacita_economico_finanziaria", {})
    req_eco = None
    fg = cef.get("fatturato_globale", {})
    if isinstance(fg, dict) and fg.get("richiesto"):
        desc_parts = []
        if fg.get("periodo_riferimento"):
            desc_parts.append(f"Fatturato globale {fg['periodo_riferimento']}")
        else:
            desc_parts.append("Fatturato globale per servizi di ingegneria e architettura")
        req_eco = RequisitiCapacitaEconomica(
            descrizione=" ".join(desc_parts) if desc_parts else None,
            importo_minimo_euro=_parse_float(fg.get("importo_minimo")),
            note=fg.get("note"),
        )

    # ── Sopralluogo ──
    sopralluogo = None
    if sop is None:
        sop = {}
    if sop:
        obblig = sop.get("obbligatorio", False)
        note_sop_parts = []
        if sop.get("modalita"):
            note_sop_parts.append(sop["modalita"])
        if sop.get("contatti_prenotazione"):
            note_sop_parts.append(f"Contatti: {sop['contatti_prenotazione']}")
        if sop.get("termine"):
            note_sop_parts.append(f"Entro il {sop['termine']}")
        if sop.get("note"):
            note_sop_parts.append(sop["note"])
        if not obblig and not note_sop_parts:
            note_sop_parts.append("Non previsto")
        sopralluogo = Sopralluogo(
            obbligatorio=obblig,
            note=". ".join(note_sop_parts) if note_sop_parts else None,
        )

    # ── Scadenza ──
    scadenza = temp.get("scadenza_offerte")

    # Add chiarimenti / apertura buste to note_particolari later
    _chiarimenti = temp.get("termine_chiarimenti") or temp.get("scadenza_chiarimenti")
    _apertura = temp.get("apertura_buste")
    _validita = temp.get("validita_offerta_giorni")

    # ── Regole presentazione offerte ──
    regole = None
    if pt:
        gestore = None
        # Cerchiamo info assistenza nel nested
        tel_ass = pt.get("telefono_assistenza")
        email_ass = pt.get("email_assistenza")
        if tel_ass or email_ass:
            gestore = GestorePiattaforma(
                telefono_assistenza=tel_ass,
                email_assistenza=email_ass,
            )
        modalita = None
        if "firma digitale" in (repr(nested).lower()):
            modalita = "Esclusivamente telematica con firma digitale"
        elif pt.get("nome"):
            modalita = "Esclusivamente telematica"

        regole = RegolePresentazioneOfferte(
            piattaforma=pt.get("url") or pt.get("nome"),
            modalita=modalita,
            gestore=gestore,
        )

    # ── Documentazione amministrativa richiesta ──
    doc_list = []
    dgue = doc.get("DGUE", {})
    if isinstance(dgue, dict) and dgue.get("richiesto"):
        firma = dgue.get("firma", "digitale")
        doc_list.append(f"DGUE compilato e firmato {firma}mente")
    anac = doc.get("contributo_ANAC", {})
    if isinstance(anac, dict) and anac.get("dovuto"):
        imp_anac = anac.get("importo_totale")
        if imp_anac:
            doc_list.append(f"Pagamento contributo ANAC (€ {imp_anac:.2f})")
        else:
            doc_list.append("Pagamento contributo ANAC")
    bollo = doc.get("imposta_bollo", {})
    if isinstance(bollo, dict) and bollo.get("importo"):
        doc_list.append(f"Pagamento imposta di bollo (€ {bollo['importo']:.2f})")
    # Soccorso istruttorio
    si = nested.get("soccorso_istruttorio", {})
    if isinstance(si, dict) and si.get("ammesso"):
        si_term = si.get("termine_giorni")
        if si_term:
            doc_list.append(f"Soccorso istruttorio ammesso (termine {si_term} giorni)")
        else:
            doc_list.append("Soccorso istruttorio ammesso")
    # Numero documenti richiesti
    n_doc = doc.get("numero_documenti_richiesti")
    if n_doc and n_doc > 3:
        doc_list.append(f"Documentazione: {n_doc} documenti richiesti")

    # ── Offerta tecnica ──
    ot_raw = cv.get("offerta_tecnica", {})
    ot_obj = None
    if ot_raw:
        criteri_out = []
        for c in ot_raw.get("criteri", []):
            if not isinstance(c, dict):
                continue
            livello = c.get("livello", "criterio")
            if livello == "sub_criterio":
                continue  # i sub vengono aggregati nei criteri padre
            sub_list = []
            codice = c.get("codice", "")
            # Cerca sub-criteri relativi a questo criterio
            for sc in ot_raw.get("criteri", []):
                if not isinstance(sc, dict):
                    continue
                sc_cod = sc.get("codice", "")
                if sc.get("livello") == "sub_criterio" and sc_cod.startswith(codice + "."):
                    sub_list.append(SubCriterio(
                        codice=sc_cod,
                        punteggio=_parse_float(sc.get("punteggio")),
                        punteggio_discrezionale=_parse_float(sc.get("punteggio_discrezionale")),
                        punteggio_tabellare=_parse_float(sc.get("punteggio_tabellare")),
                        tipo=sc.get("tipo"),
                        descrizione=sc.get("nome") or sc.get("descrizione"),
                        descrizione_dettagliata=sc.get("descrizione_dettagliata"),
                    ))
            criteri_out.append(CriterioValutazione(
                codice=codice,
                nome=c.get("nome") or c.get("descrizione"),
                punteggio=_parse_float(c.get("punteggio")),
                punteggio_discrezionale=_parse_float(c.get("punteggio_discrezionale")),
                punteggio_tabellare=_parse_float(c.get("punteggio_tabellare")),
                tipo=c.get("tipo"),
                descrizione=c.get("descrizione"),
                descrizione_dettagliata=c.get("descrizione_dettagliata"),
                sub_criteri=sub_list,
            ))

        # Formato relazione
        formato_parts = []
        if otf.get("pagine_massime"):
            fmt_pg = otf.get("formato_pagina", "A4")
            formato_parts.append(f"Max {otf['pagine_massime']} pagine ({fmt_pg})")
        if otf.get("carattere"):
            formato_parts.append(otf["carattere"])
        if otf.get("interlinea"):
            formato_parts.append(f"interlinea {otf['interlinea']}")
        if otf.get("esclusi_conteggio"):
            formato_parts.append(f"Esclusi: {otf['esclusi_conteggio']}")
        formato_rel = ", ".join(formato_parts) if formato_parts else None

        ot_obj = OffertaTecnica(
            punteggio_massimo=_parse_float(ot_raw.get("punteggio_massimo")),
            formato_relazione=formato_rel,
            contenuto_busta_tecnica=otf.get("contenuto_busta_tecnica", []),
            note_importanti=otf.get("note_importanti", []),
            criteri=criteri_out,
        )

    # ── Offerta economica ──
    oe_raw = cv.get("offerta_economica", {})
    oe_obj = None
    if oe_raw:
        mod = oe_raw.get("modalita_offerta")
        if mod == "ribasso_percentuale":
            modalita_str = "Ribasso percentuale"
            dec = oe_raw.get("cifre_decimali")
            if dec:
                modalita_str += f" (max {dec} cifre decimali)"
        elif mod:
            modalita_str = mod
        else:
            modalita_str = None

        oe_obj = OffertaEconomica(
            punteggio_massimo=_parse_float(oe_raw.get("punteggio_massimo")),
            modalita=modalita_str,
            note=oe_raw.get("formula"),
        )

    # ── Criteri valutazione offerta tecnica ──
    crit_val = None
    pt_tecnica = _parse_float(ot_raw.get("punteggio_massimo")) if ot_raw else None
    pt_economica = _parse_float(oe_raw.get("punteggio_massimo")) if oe_raw else None
    if pt_tecnica or pt_economica:
        totale = (pt_tecnica or 0) + (pt_economica or 0)

        metodo_tec = None
        if ot_raw.get("riparametrazione"):
            metodo_tec = "Con riparametrazione"

        metodo_eco = None
        formula = oe_raw.get("formula") if oe_raw else None
        if formula:
            metodo_eco = formula

        crit_val = CriteriValutazione(
            punteggio_totale=totale if totale > 0 else None,
            ripartizione=RipartizioneValutazione(
                offerta_tecnica=pt_tecnica,
                offerta_economica=pt_economica,
            ),
            metodo_valutazione=MetodoValutazione(
                tecnica=metodo_tec,
                economica=metodo_eco,
            ) if (metodo_tec or metodo_eco) else None,
        )

    # ── Vincoli partecipazione ──
    vincoli = None
    agg_max = agg.get("numero_lotti_massimi_per_concorrente")
    n_lotti = sl.get("numero_lotti", 1)
    if n_lotti > 1 or agg_max:
        vincolo_agg = None
        if agg_max:
            vincolo_agg = f"Ciascun concorrente può risultare aggiudicatario di max {agg_max} lotto/i"
        vincoli = VincoliPartecipazione(
            vincolo_partecipazione_entrambi_lotti=(n_lotti == 2),
            offerta_identica_per_entrambi_lotti=bool(sl.get("offerta_identica")),
            medesima_forma_giuridica=bool(sl.get("medesima_forma_giuridica")),
            vincolo_aggiudicazione=vincolo_agg,
        )

    # ── Tempistiche esecuzione ──
    dur = nested.get("durata_contratto", {})
    temp_exec = None
    giorni_dur = None
    note_temp = None
    if isinstance(dur, dict):
        giorni_dur = dur.get("giorni") or dur.get("durata_totale_giorni")
        # Se non abbiamo giorni, convertiamo da mesi o anni
        if not giorni_dur:
            mesi = dur.get("mesi") or dur.get("durata_totale_mesi")
            anni = dur.get("anni") or dur.get("durata_totale_anni")
            if mesi:
                giorni_dur = mesi * 30
                note_temp = f"Durata: {mesi} mesi"
            elif anni:
                giorni_dur = anni * 365
                note_temp = f"Durata: {anni} anni"
        # Fasi
        fasi = []
        for fase_key in ["fase_progettazione_mesi", "fase_realizzazione_mesi", "fase_esecuzione_lavori_mesi", "fase_gestione_mesi", "fase_gestione_anni"]:
            v = dur.get(fase_key)
            if v:
                label = fase_key.replace("_", " ").replace("fase ", "").replace(" mesi", "").replace(" anni", "").title()
                unit = "anni" if "anni" in fase_key else "mesi"
                fasi.append(f"{label}: {v} {unit}")
        if fasi:
            note_temp = (note_temp + ". " if note_temp else "") + ". ".join(fasi)
        # School year duration
        _aa_ss = dur.get("anni_scolastici")
        if _aa_ss:
            aa_str = f"Anni scolastici: {', '.join(_aa_ss)}"
            note_temp = (note_temp + ". " if note_temp else "") + aa_str
        if dur.get("prorogabile") is False:
            note_temp = (note_temp + " – " if note_temp else "") + "non prorogabile"

    durata_gg = temp.get("durata_esecuzione_giorni") or giorni_dur
    # Fallback da mesi nelle tempistiche
    if not durata_gg and temp.get("durata_esecuzione_mesi"):
        durata_gg = temp["durata_esecuzione_mesi"] * 30
        note_temp = f"Durata: {temp['durata_esecuzione_mesi']} mesi"

    if durata_gg or note_temp:
        temp_exec = TempistichEsecuzione(
            durata_complessiva_stimata_giorni=durata_gg,
            note=note_temp,
        )

    # ── Garanzia provvisoria ──
    gp_raw = gar.get("garanzia_provvisoria", {})
    gd_raw = gar.get("garanzia_definitiva", {})
    pol_raw = gar.get("polizza_RC_professionale", {})

    gar_prov = None
    if isinstance(gp_raw, dict) and any(k in gp_raw for k in ("dovuta", "percentuale", "importo")):
        gar_prov = GaranziaProvvisoria(
            richiesta=gp_raw.get("dovuta", False),
            percentuale=_parse_float(gp_raw.get("percentuale")),
            importo=_parse_float(gp_raw.get("importo")),
            note=gp_raw.get("nota"),
        )

    # ── Garanzia definitiva ──
    gar_def = None
    if isinstance(gd_raw, dict) and any(k in gd_raw for k in ("dovuta", "percentuale", "note")):
        gar_def = GaranziaDefinitiva(
            richiesta=gd_raw.get("dovuta", False),
            percentuale=_parse_float(gd_raw.get("percentuale")),
            note=gd_raw.get("note"),
        )

    # ── Polizza RC professionale ──
    pol_rc = None
    if isinstance(pol_raw, dict) and pol_raw.get("richiesta"):
        pol_rc = PolizzaRC(
            richiesta=True,
            copertura=pol_raw.get("copertura"),
        )

    # ── Revisione prezzi ──
    rev_raw = ic.get("revisione_prezzi", {})
    rev = None
    if isinstance(rev_raw, dict):
        rev = RevisionePrezzi(
            ammessa=rev_raw.get("ammessa", False),
            soglia_percentuale=_parse_float(rev_raw.get("soglia_percentuale")),
            note=rev_raw.get("note"),
        )

    # ── Note particolari ──
    note_list = []
    if n_lotti > 1:
        note_list.append(f"Procedura suddivisa in {n_lotti} lotti")
    if tp.get("inversione_procedimentale"):
        note_list.append("Possibile inversione procedimentale ex art. 107 comma 3")
    sic = nested.get("sicurezza", {})
    duvri = sic.get("DUVRI", {}) if isinstance(sic, dict) else {}
    if isinstance(duvri, dict) and duvri.get("richiesto") is False:
        nota_duvri = duvri.get("nota", "")
        note_list.append(f"DUVRI non previsto{': ' + nota_duvri if nota_duvri else ''}")
    elif isinstance(duvri, dict) and duvri.get("richiesto"):
        note_list.append("DUVRI richiesto")
    # Subappalto
    sub = nested.get("subappalto", {})
    if isinstance(sub, dict):
        if sub.get("ammesso"):
            limite = sub.get("limite_percentuale")
            note_list.append(f"Subappalto ammesso{f' (max {limite}%)' if limite else ''}")
        elif sub.get("ammesso") is False:
            note_list.append("Subappalto non ammesso")
    # Avvalimento
    avv_nested = nested.get("avvalimento", {})
    if isinstance(avv_nested, dict):
        if avv_nested.get("ammesso"):
            note_list.append("Avvalimento ammesso")
        elif avv_nested.get("ammesso") is False:
            note_list.append("Avvalimento non ammesso")
    # Penali
    pen = nested.get("penali", {})
    if isinstance(pen, dict) and pen.get("previste"):
        perc_pen = pen.get("percentuale_giornaliera")
        note_list.append(f"Penali previste{f': {perc_pen}% giornaliero' if perc_pen else ''}")
    # Controversie / Foro
    cont = nested.get("controversie", {})
    if isinstance(cont, dict):
        foro = cont.get("foro_competente")
        if foro:
            note_list.append(f"Foro competente: {foro}")
    # Aggiudicazione
    stip_contr = agg.get("stipula_contratto", {})
    if isinstance(stip_contr, dict) and stip_contr.get("forma"):
        dettaglio = stip_contr.get("dettaglio")
        if dettaglio:
            note_list.append(f"Stipula: {dettaglio}")
        else:
            note_list.append(f"Stipula: {stip_contr['forma'].replace('_', ' ')}")
    # Finanziamento
    fin = ig.get("finanziamento", {})
    if isinstance(fin, dict) and fin.get("fonte"):
        note_list.append(f"Finanziamento: {fin['fonte']}")
    # Fonti finanziamento da info_aggiuntive
    fonti = ia.get("fonti_finanziamento", [])
    if fonti and not fin.get("fonte"):
        note_list.append(f"Finanziamento: {', '.join(fonti)}")
    # Oneri sicurezza
    oneri = ic.get("oneri_sicurezza")
    if oneri:
        note_list.append(f"Oneri sicurezza: € {oneri:,.2f}")    # CCNL
    _ccnl_note = None
    rp_raw = nested.get("requisiti_partecipazione", {})
    ctp_raw = rp_raw.get("capacita_tecnico_professionale", {}) if isinstance(rp_raw, dict) else {}
    _srv_raw = ctp_raw.get("servizi_analoghi", {}) if isinstance(ctp_raw, dict) else {}
    _ccnl_note = ctp_raw.get("ccnl") if isinstance(ctp_raw, dict) else None
    if _ccnl_note:
        note_list.append(f"CCNL applicabile: {_ccnl_note}")
    # Clausola sociale
    if isinstance(_srv_raw, dict) and _srv_raw.get("clausola_sociale"):
        note_list.append("Clausola sociale: obbligo di riassorbimento personale")
    # Quote occupazionali
    _quote_raw = _srv_raw.get("quote_occupazionali", {}) if isinstance(_srv_raw, dict) else {}
    if isinstance(_quote_raw, dict):
        if _quote_raw.get("giovani_30_percento"):
            note_list.append("Riserva assunzioni: 30% a giovani")
        if _quote_raw.get("donne_30_percento"):
            note_list.append("Riserva assunzioni: 30% a donne")    # Cause esclusione
    ce = nested.get("cause_esclusione", {})
    if isinstance(ce, dict):
        if ce.get("automatiche"):
            rif = ce["automatiche"].get("riferimento", "") if isinstance(ce["automatiche"], dict) else ""
            note_list.append(f"Cause esclusione automatiche {rif}".strip())
    # Accordo quadro
    aq_tp = tp.get("accordo_quadro", {})
    if isinstance(aq_tp, dict) and aq_tp.get("presente"):
        dur_aq = aq_tp.get("durata_mesi")
        note_list.append(f"Accordo quadro{f' ({dur_aq} mesi)' if dur_aq else ''}")
    # Note dall'info_aggiuntive
    ia_note = ia.get("note", [])
    if isinstance(ia_note, list):
        note_list.extend(ia_note)
    # Tempistiche aggiuntive
    if _chiarimenti:
        note_list.append(f"Termine chiarimenti: {_chiarimenti}")
    if _apertura:
        note_list.append(f"Apertura buste: {_apertura}")
    if _validita:
        note_list.append(f"Validità offerta: {_validita} giorni")

    # ── CAM ──
    cam = nested.get("CAM_criteri_ambientali", {})
    cam_str = None
    if isinstance(cam, dict) and cam.get("applicabili"):
        decreto = cam.get("decreto_riferimento", "")
        cam_str = f"Conformità ai CAM {decreto}".strip()

    # ── Costruzione modello ──
    output = AppaltoOutput(
        cig=cig_dict,
        cup=cup,
        codice_NUTS=ig.get("codice_NUTS"),
        CPV_principale=ig.get("CPV_principale"),
        procedura_tipo=tipo_proc or None,
        criterio_aggiudicazione=criterio or None,
        oggetto_appalto=oggetto,
        descrizione_lavori_con_importo_totale=descr_lavori,
        stazione_appaltante=sa,
        tipologia_appalto=tipologia_appalto,
        requisiti_idoneita_professionale=req_idon,
        requisiti_capacita_tecnica_professionale=req_tec,
        requisiti_capacita_economica_finanziaria=req_eco,
        sopralluogo=sopralluogo,
        scadenza=scadenza,
        regole_presentazione_offerte=regole,
        documentazione_amministrativa_richiesta=doc_list,
        offerta_tecnica=ot_obj,
        offerta_economica=oe_obj,
        criteri_valutazione_offerta_tecnica=crit_val,
        vincoli_partecipazione=vincoli,
        tempistiche_esecuzione=temp_exec,
        garanzia_provvisoria=gar_prov,
        garanzia_definitiva=gar_def,
        polizza_RC_professionale=pol_rc,
        CAM_criteri_ambientali=cam_str,
        revisione_prezzi=rev,
        note_particolari=note_list,
    )

    return output.model_dump(exclude_none=True)


def build_output_with_methods(nested: dict) -> tuple[dict, dict]:
    """
    Come build_output ma restituisce anche il dizionario methods
    (quale campo è stato estratto con quale metodo).

    Returns:
        (output_dict, methods_dict)
    """
    output = build_output(nested)
    methods = {}
    for key, val in output.items():
        if val is not None and val != [] and val != {}:
            methods[key] = "rules"
    return output, methods
