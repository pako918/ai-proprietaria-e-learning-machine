"""
Flatten – Converte il JSON nested profondo in formato piatto per pipeline/UI.
"""

from __future__ import annotations


def _format_euro(val) -> str | None:
    """Formatta un importo numerico come stringa Euro leggibile."""
    if val is None:
        return None
    try:
        num = float(val)
        if num <= 0:
            return None
        formatted = f"{num:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"EUR {formatted}"
    except (ValueError, TypeError):
        return str(val) if val else None


def flatten_for_pipeline(nested: dict) -> tuple[dict, dict, dict]:
    """
    Converte il JSON nested profondo di extract_rules_based()
    nel formato piatto usato dalla pipeline e dalla UI.

    Returns:
        (flat_result, snippets, methods)
    """
    flat: dict = {}
    methods: dict = {}
    snippets: dict = {}

    def _g(*keys):
        """Navigate nested dict with fallback."""
        obj = nested
        for k in keys:
            if isinstance(obj, dict):
                obj = obj.get(k)
            else:
                return None
        return obj

    # ── Identificativi ──
    ig = nested.get("informazioni_generali", {})
    sa = ig.get("stazione_appaltante", {}) if isinstance(ig.get("stazione_appaltante"), dict) else {}
    rup_d = ig.get("RUP", ig.get("rup", {}))

    flat["numero_bando"] = ig.get("numero_bando")
    flat["cig"] = ig.get("CIG")
    flat["cup"] = ig.get("CUP")
    flat["cui"] = ig.get("CUI")
    flat["cpv_principale"] = ig.get("CPV_principale")
    flat["cpv_secondari"] = ig.get("CPV_secondari", [])
    flat["codice_nuts"] = ig.get("codice_NUTS")

    det = ig.get("determina_a_contrarre", ig.get("determina", {}))
    if isinstance(det, dict):
        num = det.get("numero", "")
        data = det.get("data", "")
        flat["determina_a_contrarre"] = f"n. {num} del {data}" if num else None
    elif det:
        flat["determina_a_contrarre"] = str(det)
    else:
        flat["determina_a_contrarre"] = None

    # ── Stazione Appaltante e RUP ──
    flat["stazione_appaltante"] = sa.get("denominazione")
    flat["tipo_ente"] = sa.get("tipo_ente")
    flat["sa_indirizzo"] = sa.get("indirizzo")
    flat["sa_pec"] = sa.get("PEC") or sa.get("pec")
    flat["sa_email"] = sa.get("email")
    flat["sa_telefono"] = sa.get("telefono")
    flat["sa_sito_web"] = sa.get("sito_web")
    flat["sa_area_direzione"] = sa.get("area_direzione")
    flat["rup"] = rup_d.get("nome") if isinstance(rup_d, dict) else rup_d
    flat["rup_qualifica"] = rup_d.get("qualifica") if isinstance(rup_d, dict) else None
    flat["rup_email"] = rup_d.get("email") if isinstance(rup_d, dict) else None
    flat["rup_pec"] = rup_d.get("PEC") if isinstance(rup_d, dict) else None

    # ── Oggetto ──
    flat["oggetto_appalto"] = ig.get("titolo")
    flat["oggetto_sintetico"] = ig.get("oggetto_sintetico")

    # ── Tipo procedura ──
    tp = nested.get("tipo_procedura", {})
    flat["tipo_procedura"] = tp.get("tipo")
    flat["ambito_procedura"] = tp.get("ambito")
    flat["criterio_aggiudicazione"] = tp.get("criterio_aggiudicazione")
    flat["metodo_oepv"] = tp.get("metodo_OEPV")
    flat["rif_normativo_procedura"] = tp.get("riferimento_normativo")
    flat["inversione_procedimentale"] = tp.get("inversione_procedimentale", False)

    aq = tp.get("accordo_quadro", {})
    if isinstance(aq, dict) and aq.get("presente"):
        aq_parts = [aq.get("tipo", "")]
        if aq.get("durata_mesi"):
            aq_parts.append(f"{aq['durata_mesi']} mesi")
        flat["accordo_quadro"] = " - ".join(filter(None, aq_parts)) or "Si"
    elif isinstance(aq, bool) and aq:
        flat["accordo_quadro"] = "Si"
    else:
        flat["accordo_quadro"] = None

    flat["concessione"] = tp.get("concessione", False)

    # ── Piattaforma Telematica ──
    pt = nested.get("piattaforma_telematica", {})
    flat["piattaforma_nome"] = pt.get("nome")
    flat["piattaforma_url"] = pt.get("url")
    flat["piattaforma_gestore"] = pt.get("gestore")

    # ── Suddivisione Lotti ──
    sl = nested.get("suddivisione_lotti", {})
    flat["numero_lotti"] = sl.get("numero_lotti")
    flat["lotto_unico_motivazione"] = sl.get("lotto_unico_motivazione")
    flat["max_lotti_aggiudicabili"] = sl.get("numero_massimo_lotti_aggiudicabili")
    flat["vincoli_lotti"] = sl.get("vincoli_partecipazione_lotti")
    flat["lotti"] = sl.get("lotti", [])

    # ── Importi Complessivi ──
    ic = nested.get("importi_complessivi", {})
    flat["importo_totale_gara"] = _format_euro(ic.get("importo_totale_gara"))
    flat["importo_soggetto_ribasso"] = _format_euro(
        ic.get("importo_totale_soggetto_ribasso") or ic.get("importo_base_gara")
    )
    flat["importo_non_soggetto_ribasso"] = _format_euro(ic.get("importo_totale_non_soggetto_ribasso"))
    flat["importo_lavori"] = _format_euro(ic.get("importo_lavori_complessivo"))
    flat["oneri_sicurezza"] = _format_euro(ic.get("oneri_sicurezza"))
    flat["costi_manodopera"] = _format_euro(ic.get("costi_manodopera"))
    flat["quota_ribassabile_percentuale"] = ic.get("quota_ribassabile_percentuale")

    ant = ic.get("anticipazione", ic.get("anticipazione_prezzo", {}))
    if isinstance(ant, dict):
        if ant.get("prevista"):
            flat["anticipazione"] = f"{ant.get('percentuale', '')}%"
        else:
            flat["anticipazione"] = None
    elif ant:
        flat["anticipazione"] = _format_euro(ant) if isinstance(ant, (int, float)) else str(ant)
    else:
        flat["anticipazione"] = None

    rev = ic.get("revisione_prezzi", {})
    if isinstance(rev, dict) and rev.get("ammessa"):
        parts = ["Ammessa"]
        if rev.get("soglia_percentuale"):
            parts.append(f"soglia {rev['soglia_percentuale']}%")
        flat["revisione_prezzi"] = " - ".join(parts)
    elif isinstance(rev, dict) and rev.get("ammessa") is False:
        flat["revisione_prezzi"] = "Non ammessa"
    elif isinstance(rev, bool):
        flat["revisione_prezzi"] = "Ammessa" if rev else None
    else:
        flat["revisione_prezzi"] = None

    # ── Durata Contratto ──
    dur = nested.get("durata_contratto", {})
    if isinstance(dur, dict):
        if dur.get("durata_totale_anni") or dur.get("anni"):
            y = dur.get("durata_totale_anni") or dur.get("anni")
            flat["durata_contratto"] = f"{y} anni"
        elif dur.get("durata_totale_mesi") or dur.get("mesi"):
            m = dur.get("durata_totale_mesi") or dur.get("mesi")
            flat["durata_contratto"] = f"{m} mesi"
        elif dur.get("durata_totale_giorni") or dur.get("giorni"):
            g = dur.get("durata_totale_giorni") or dur.get("giorni")
            flat["durata_contratto"] = f"{g} giorni"
        else:
            flat["durata_contratto"] = None
        flat["decorrenza"] = dur.get("decorrenza")
        fp = dur.get("fase_progettazione_mesi") or dur.get("fase_realizzazione_mesi")
        flat["fase_progettazione"] = f"{fp} mesi" if fp else None
        fl = dur.get("fase_esecuzione_lavori_mesi")
        flat["fase_lavori"] = f"{fl} mesi" if fl else None
        fg_a = dur.get("fase_gestione_anni")
        fg_m = dur.get("fase_gestione_mesi")
        if fg_a:
            flat["fase_gestione"] = f"{fg_a} anni"
        elif fg_m:
            flat["fase_gestione"] = f"{fg_m} mesi"
        else:
            flat["fase_gestione"] = None
        pr = dur.get("proroga", {})
        if isinstance(pr, dict) and pr.get("ammessa"):
            flat["proroga"] = f"Ammessa - {pr.get('durata_mesi', '')} mesi" if pr.get("durata_mesi") else "Ammessa"
        else:
            flat["proroga"] = None
        sds = dur.get("societa_di_scopo", {})
        if isinstance(sds, dict) and sds.get("obbligatoria"):
            flat["societa_di_scopo"] = sds.get("forma") or "Obbligatoria"
        else:
            flat["societa_di_scopo"] = None
    else:
        flat["durata_contratto"] = str(dur) if dur else None
        flat["decorrenza"] = None
        flat["fase_progettazione"] = None
        flat["fase_lavori"] = None
        flat["fase_gestione"] = None
        flat["proroga"] = None
        flat["societa_di_scopo"] = None

    # ── Tempistiche ──
    temp = nested.get("tempistiche", {})
    flat["scadenza_offerte"] = temp.get("scadenza_offerte")
    flat["apertura_buste"] = temp.get("apertura_buste")
    flat["scadenza_chiarimenti"] = temp.get("termine_chiarimenti") or temp.get("scadenza_chiarimenti")
    flat["validita_offerta_giorni"] = temp.get("validita_offerta_giorni")
    flat["termine_stipula_giorni"] = temp.get("termine_stipula_giorni")
    agg_raw = nested.get("aggiudicazione", {})
    stip_raw = agg_raw.get("stipula_contratto", {}) if isinstance(agg_raw, dict) else {}
    flat["standstill_giorni"] = (
        temp.get("standstill_giorni")
        or (stip_raw.get("termine_standstill_giorni") if isinstance(stip_raw, dict) else None)
    )

    # ── Requisiti Partecipazione ──
    rp = nested.get("requisiti_partecipazione", {})
    sogg = rp.get("soggetti_ammessi", {})
    flat["obbligo_giovane"] = sogg.get("obbligo_giovane_professionista", False) if isinstance(sogg, dict) else False

    gdl = rp.get("gruppo_di_lavoro", {})
    fig_list = gdl.get("figure_professionali", [])
    flat["numero_figure_richieste"] = len(fig_list) if fig_list else None
    flat["figure_professionali"] = [
        f.get("ruolo", "") + (f" ({f['requisiti']})" if f.get("requisiti") else "")
        for f in fig_list if isinstance(f, dict) and f.get("ruolo")
    ] if fig_list else []
    flat["ruoli_cumulabili"] = gdl.get("ruoli_cumulabili", False)

    cef = rp.get("capacita_economico_finanziaria", {})
    fg = cef.get("fatturato_globale", cef.get("fatturato_globale_minimo", {}))
    if isinstance(fg, dict):
        flat["fatturato_globale_minimo"] = _format_euro(fg.get("importo_minimo"))
        flat["fatturato_periodo"] = fg.get("periodo_riferimento")
    elif fg:
        flat["fatturato_globale_minimo"] = _format_euro(fg)
        flat["fatturato_periodo"] = None
    else:
        flat["fatturato_globale_minimo"] = None
        flat["fatturato_periodo"] = cef.get("periodo_riferimento_anni")

    fs = cef.get("fatturato_specifico", cef.get("fatturato_specifico_minimo", {}))
    flat["fatturato_specifico_minimo"] = _format_euro(fs.get("importo_minimo") if isinstance(fs, dict) else fs) if fs else None

    cop = cef.get("copertura_assicurativa", {})
    flat["copertura_assicurativa"] = _format_euro(cop.get("importo_minimo") if isinstance(cop, dict) else cop) if cop else None

    # ── Categorie e SOA ──
    ctp = rp.get("capacita_tecnico_professionale", {})

    soa = ctp.get("attestazione_SOA", ctp.get("requisiti_SOA", {}))
    if isinstance(soa, dict):
        flat["soa_richiesta"] = soa.get("richiesta", False)
        soa_cats = soa.get("categorie", [])
        flat["soa_categorie"] = [
            f"{c.get('id_categoria', c.get('categoria', ''))} - cl. {c.get('classifica', c.get('tipo', ''))}"
            for c in soa_cats if isinstance(c, dict)
        ] if soa_cats else []
    elif soa:
        flat["soa_richiesta"] = True
        flat["soa_categorie"] = [str(soa)] if not isinstance(soa, list) else soa
    else:
        flat["soa_richiesta"] = False
        flat["soa_categorie"] = []

    raw_cats = ig.get("categorie_trovate", [])
    formatted_cats = []
    for c in raw_cats:
        if isinstance(c, dict):
            cat_id = c.get("id_categoria", c.get("categoria", ""))
            desc = c.get("descrizione", "")
            imp = c.get("importo_opera", c.get("importo", 0))
            imp_str = _format_euro(imp) if imp else ""
            parts = [cat_id]
            if desc:
                parts.append(desc[:80])
            if imp_str:
                parts.append(imp_str)
            formatted_cats.append(" - ".join(parts))
        elif isinstance(c, str):
            formatted_cats.append(c)
    flat["categorie_opere"] = formatted_cats

    # ── Criteri Valutazione ──
    cv = nested.get("criteri_valutazione", {})
    ot = cv.get("offerta_tecnica", {})
    oe = cv.get("offerta_economica", {})

    flat["punteggio_tecnica"] = ot.get("punteggio_massimo") or ot.get("punteggio_max")
    flat["punteggio_economica"] = oe.get("punteggio_massimo") or oe.get("punteggio_max")
    flat["soglia_sbarramento"] = ot.get("soglia_sbarramento")
    flat["riparametrazione"] = ot.get("riparametrazione", False)
    flat["formula_economica"] = oe.get("formula")
    flat["modalita_offerta"] = oe.get("modalita_offerta")
    flat["cifre_decimali"] = oe.get("cifre_decimali")

    criteri_list = ot.get("criteri", [])
    flat["criteri_tecnici"] = [
        {"codice": c.get("codice", ""), "nome": c.get("descrizione", c.get("nome", "")), "punteggio": c.get("punteggio", 0)}
        for c in criteri_list
    ] if criteri_list else []

    va = cv.get("verifica_anomalia", {})
    flat["verifica_anomalia"] = va.get("prevista") if isinstance(va, dict) else bool(va)

    # ── Offerta Tecnica Formato ──
    otf = nested.get("offerta_tecnica_formato", {})
    flat["ot_formato_pagina"] = otf.get("formato_pagina")
    flat["ot_carattere"] = otf.get("carattere")
    flat["ot_interlinea"] = otf.get("interlinea")
    flat["ot_limite_pagine"] = otf.get("pagine_massime") or otf.get("limite_pagine_totale")

    # ── Garanzie ──
    gar = nested.get("garanzie", {})
    gp = gar.get("garanzia_provvisoria", {})
    gd = gar.get("garanzia_definitiva", {})
    pol = gar.get("polizza_RC_professionale", gar.get("polizza_professionale", {}))

    flat["gar_provvisoria_dovuta"] = gp.get("dovuta") if isinstance(gp, dict) else None
    flat["gar_provvisoria_percentuale"] = f"{gp['percentuale']}%" if isinstance(gp, dict) and gp.get("percentuale") else None
    flat["gar_provvisoria_importo"] = _format_euro(gp.get("importo")) if isinstance(gp, dict) else None
    flat["gar_provvisoria_durata"] = gp.get("durata_giorni") if isinstance(gp, dict) else None
    flat["gar_definitiva_percentuale"] = f"{gd['percentuale']}%" if isinstance(gd, dict) and gd.get("percentuale") else None
    flat["gar_definitiva_forma"] = gd.get("forma") if isinstance(gd, dict) else None
    flat["polizza_rc"] = pol.get("richiesta", False) if isinstance(pol, dict) else bool(pol)
    flat["polizza_rc_copertura"] = pol.get("copertura") if isinstance(pol, dict) else None

    # ── Subappalto ──
    sub = nested.get("subappalto", {})
    flat["subappalto_ammesso"] = sub.get("ammesso") if isinstance(sub, dict) else None
    flat["subappalto_limite"] = f"{sub['limite_percentuale']}%" if isinstance(sub, dict) and sub.get("limite_percentuale") else None
    flat["subappalto_condizioni"] = sub.get("condizioni") if isinstance(sub, dict) else None

    # ── Avvalimento ──
    avv = nested.get("avvalimento", {})
    flat["avvalimento_ammesso"] = avv.get("ammesso") if isinstance(avv, dict) else None
    flat["avvalimento_condizioni"] = avv.get("condizioni") if isinstance(avv, dict) else None

    # ── Sopralluogo ──
    sop = nested.get("sopralluogo", {})
    flat["sopralluogo_obbligatorio"] = sop.get("obbligatorio", False) if isinstance(sop, dict) else bool(sop)
    flat["sopralluogo_modalita"] = sop.get("modalita") if isinstance(sop, dict) else None
    flat["sopralluogo_contatti"] = sop.get("contatti_prenotazione") or (sop.get("contatti") if isinstance(sop, dict) else None)
    flat["sopralluogo_termine"] = sop.get("termine") if isinstance(sop, dict) else None

    # ── Documentazione Amministrativa ──
    doc = nested.get("documentazione_amministrativa", {})
    dgue = doc.get("DGUE", {})
    flat["dgue_richiesto"] = dgue.get("richiesto", False) if isinstance(dgue, dict) else bool(dgue)
    anac = doc.get("contributo_ANAC", {})
    flat["contributo_anac"] = _format_euro(anac.get("importo_totale") or anac.get("importo") if isinstance(anac, dict) else anac)
    bollo = doc.get("imposta_bollo", {})
    flat["imposta_bollo"] = _format_euro(bollo.get("importo") if isinstance(bollo, dict) else bollo)
    flat["numero_documenti"] = doc.get("numero_documenti_richiesti")

    # ── Soccorso Istruttorio ──
    socc = nested.get("soccorso_istruttorio", {})
    flat["soccorso_ammesso"] = socc.get("ammesso", socc.get("previsto", False)) if isinstance(socc, dict) else bool(socc)
    flat["soccorso_termine_giorni"] = socc.get("termine_giorni", socc.get("termine")) if isinstance(socc, dict) else None
    flat["soccorso_riferimento"] = socc.get("riferimento") if isinstance(socc, dict) else None

    # ── Cause Esclusione ──
    ce = nested.get("cause_esclusione", {})
    auto_ce = ce.get("automatiche", {})
    flat["esclusione_automatiche"] = auto_ce.get("riferimento") if isinstance(auto_ce, dict) else (str(auto_ce) if auto_ce else None)
    nauto_ce = ce.get("non_automatiche", {})
    flat["esclusione_non_automatiche"] = nauto_ce.get("riferimento") if isinstance(nauto_ce, dict) else (str(nauto_ce) if nauto_ce else None)
    sc = ce.get("self_cleaning", {})
    flat["self_cleaning"] = sc.get("ammesso", False) if isinstance(sc, dict) else bool(sc)

    # ── Aggiudicazione e Stipula ──
    agg = nested.get("aggiudicazione", {})
    flat["max_lotti_concorrente"] = agg.get("numero_lotti_massimi_per_concorrente") or agg.get("max_lotti_per_concorrente")
    stip = agg.get("stipula_contratto", {})
    flat["stipula_forma"] = stip.get("forma") if isinstance(stip, dict) else None
    flat["stipula_standstill"] = stip.get("termine_standstill_giorni") if isinstance(stip, dict) else None
    flat["esecuzione_anticipata"] = stip.get("esecuzione_anticipata", False) if isinstance(stip, dict) else False

    # ── Penali ──
    pen = nested.get("penali", {})
    flat["penali_previste"] = pen.get("previste", False) if isinstance(pen, dict) else bool(pen)
    flat["penali_percentuale"] = f"{pen['percentuale_giornaliera']}%" if isinstance(pen, dict) and pen.get("percentuale_giornaliera") else None
    flat["penali_tetto"] = f"{pen['tetto_massimo_percentuale']}%" if isinstance(pen, dict) and pen.get("tetto_massimo_percentuale") else None
    flat["penali_descrizione"] = pen.get("descrizione") or pen.get("ritardo") if isinstance(pen, dict) else None

    # ── Sicurezza ──
    sic = nested.get("sicurezza", {})
    flat["sicurezza_oneri"] = _format_euro(sic.get("oneri_sicurezza_interferenza") or sic.get("oneri_sicurezza"))
    duvri = sic.get("DUVRI", {})
    flat["duvri_richiesto"] = duvri.get("richiesto") if isinstance(duvri, dict) else None
    flat["duvri_nota"] = duvri.get("nota") if isinstance(duvri, dict) else None

    # ── CAM Criteri Ambientali ──
    cam = nested.get("CAM_criteri_ambientali", {})
    flat["cam_applicabili"] = cam.get("applicabili", False) if isinstance(cam, dict) else bool(cam)
    flat["cam_decreto"] = cam.get("decreto_riferimento") if isinstance(cam, dict) else None
    flat["cam_requisiti"] = cam.get("requisiti_minimi") if isinstance(cam, dict) else None

    # ── Controversie ──
    cont = nested.get("controversie", {})
    flat["foro_competente"] = cont.get("foro_competente")
    flat["termine_ricorso_giorni"] = cont.get("termine_ricorso_giorni")
    flat["arbitrato"] = cont.get("arbitrato")
    cct = cont.get("collegio_consultivo_tecnico", {})
    flat["collegio_consultivo"] = cct.get("previsto", False) if isinstance(cct, dict) else bool(cct)

    # ── Tracciabilita e Info Aggiuntive ──
    trac = nested.get("tracciabilita_flussi", {})
    flat["tracciabilita_flussi"] = trac.get("obbligatoria", False) if isinstance(trac, dict) else bool(trac)

    ia = nested.get("informazioni_aggiuntive", {})
    flat["natura_servizio"] = ia.get("natura_servizio")
    flat["codice_comportamento"] = ia.get("codice_comportamento", False)
    flat["finanziamento"] = _g("informazioni_generali", "finanziamento", "fonte")
    flat["fonti_finanziamento"] = ia.get("fonti_finanziamento", [])
    flat["ccnl"] = ia.get("CCNL")

    comm = ia.get("commissione_giudicatrice", {})
    if isinstance(comm, dict) and comm.get("prevista"):
        flat["commissione_giudicatrice"] = f"Si - {comm.get('numero_membri', '?')} membri"
    else:
        flat["commissione_giudicatrice"] = None

    pnrr = ia.get("condizioni_PNRR", ia.get("condizioni_occupazionali_PNRR", {}))
    if isinstance(pnrr, dict):
        flat["pnrr_giovani"] = pnrr.get("quota_min_occupazione_giovanile_percentuale")
        flat["pnrr_donne"] = pnrr.get("quota_min_occupazione_femminile_percentuale")
        flat["dnsh"] = pnrr.get("DNSH", False)
    else:
        flat["pnrr_giovani"] = None
        flat["pnrr_donne"] = None
        flat["dnsh"] = False

    # ── Note operative (auto-generate) ──
    note: list[str] = []
    if flat.get("sopralluogo_obbligatorio"):
        note.append("Sopralluogo obbligatorio")
    if flat.get("inversione_procedimentale"):
        note.append("Inversione procedimentale (art. 107 c.3)")
    if flat.get("soglia_sbarramento"):
        note.append(f"Soglia sbarramento tecnica: {flat['soglia_sbarramento']} punti")
    if flat.get("cam_applicabili"):
        note.append("Conformita CAM obbligatoria")
    if flat.get("finanziamento"):
        note.append(f"Finanziamento: {flat['finanziamento']}")
    if flat.get("soccorso_ammesso"):
        note.append("Soccorso istruttorio previsto")
    if flat.get("dnsh"):
        note.append("Principio DNSH (PNRR)")
    if flat.get("penali_previste"):
        note.append("Penali previste")
    if flat.get("subappalto_ammesso"):
        note.append("Subappalto ammesso")
    flat["note_operative"] = note

    # Build methods
    for k, v in flat.items():
        if v is not None and v != "" and v != [] and v != {} and v is not False and v != 0:
            methods[k] = "rules"

    flat["_nested_full"] = nested

    return flat, snippets, methods
