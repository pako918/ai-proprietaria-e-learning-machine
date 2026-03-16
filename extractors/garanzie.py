"""
Sezione I – Garanzie.

Estrae garanzia provvisoria, garanzia definitiva e polizza RC professionale.
"""

from __future__ import annotations
import re
from .utils import _clean, _parse_euro, _section_text


def extract_garanzie(text: str, text_lower: str) -> dict:
    """
    Restituisce il dict ``garanzie`` con sotto-chiavi:

    * garanzia_provvisoria
    * garanzia_definitiva
    * polizza_RC_professionale
    """

    gar: dict = {"garanzia_provvisoria": {}, "garanzia_definitiva": {}, "polizza_RC_professionale": {}}
    gp = gar["garanzia_provvisoria"]
    gd = gar["garanzia_definitiva"]
    pol = gar["polizza_RC_professionale"]

    # ── Garanzia provvisoria ─────────────────────────────────────────────
    gar_section = _section_text(
        text,
        ["GARANZIA PROVVISORIA", "10. GARANZI"],
        ["GARANZIA DEFINITIVA", "11.", "SOPRALLUOGO"],
        max_len=5000,
    )

    _gar_check = gar_section.lower() if gar_section.strip() else text_lower
    _gar_non_dovuta = (
        "non dovuta" in _gar_check
        or "non è dovuta" in _gar_check
        or "non richiesta" in _gar_check
        or "non è richiesta" in _gar_check
    )
    if not _gar_non_dovuta and not gar_section.strip():
        _gar_non_dovuta = bool(re.search(
            r"non\s+è\s+richiest\w+\s+la\s+garanzia\s+provvisoria",
            text, re.IGNORECASE,
        ))
    if _gar_non_dovuta:
        gp["dovuta"] = False
        m_gar_ref = re.search(
            r"(?:ai\s+sensi\s+dell.art\.?\s*\d+\s+comma\s*\d+(?:[^.]|\.\s*(?=\d)){0,60}non\s+.{0,30}richiest\w*[^.]*\.|"
            r"non\s+.{0,15}richiest\w*.{0,60}garanzia\s+provvisoria[^.]*art\.?\s*\d+[^.]*\.)",
            text, re.IGNORECASE,
        )
        if m_gar_ref:
            gp["nota"] = _clean(m_gar_ref.group(0))[:300]
    else:
        gp_perc_patterns = [
            r"(\d+(?:[.,]\d+)?)\s*%\s*(?:dell?\s*['\u2019]?\s*import)",
            r"pari\s+(?:al?\s+)?(\d+(?:[.,]\d+)?)\s*%",
            r"misura\s+(?:del?\s+)?(\d+(?:[.,]\d+)?)\s*%",
        ]
        for _gpp in gp_perc_patterns:
            m_gp = re.search(_gpp, gar_section, re.IGNORECASE)
            if m_gp:
                _gpval = float(m_gp.group(1).replace(",", "."))
                if 0.5 <= _gpval <= 10:
                    gp["dovuta"] = True
                    gp["percentuale"] = _gpval
                    break
        if "percentuale" not in gp and re.search(r"garanzia\s+provvisoria", text, re.IGNORECASE):
            gp["dovuta"] = True

    # Importo garanzia provvisoria
    gp_imp_patterns = [
        r"(?:garanzia\s+provvisoria|cauzione)[^€Ç\d]{0,200}?(?:importo\s*)?(?:pari\s+ad?\s+)?(?:€|Ç|euro)\s*\.?\s*([\d.,]+)",
        r"(?:garanzia\s+provvisoria|cauzione)[^€Ç\d]{0,80}?(?:€|Ç|\s)([\d]{1,3}(?:[.,]\d{3})*[.,]\d{2})",
        r"(?:garanzia\s+provvisoria|cauzione)[^€Ç\d]{0,80}?[€Ç\s]*([\d.,]+)",
        r"(?:importo|pari\s+ad?)\s+(?:€|Ç|euro)\s*\.?\s*([\d.,]+)",
    ]
    for gp_pat in gp_imp_patterns:
        m_gp_imp = re.search(gp_pat, gar_section, re.IGNORECASE | re.DOTALL)
        if m_gp_imp:
            v = _parse_euro(m_gp_imp.group(1))
            if v and v > 100:
                gp["importo"] = v
                break

    # ── Garanzia definitiva ──────────────────────────────────────────────
    gd_section = _section_text(
        text,
        ["GARANZIA DEFINITIVA"],
        ["11.", "SOPRALLUOGO", "12.", "PAGAMENTO", "CONTRIBUTO"],
        max_len=3000,
    )
    m_gd = re.search(r"(\d+(?:[.,]\d+)?)\s*%\s*(?:dell?\s*['\u2019]?\s*import)", gd_section, re.IGNORECASE)
    if m_gd:
        gd["dovuta"] = True
        gd["percentuale"] = float(m_gd.group(1).replace(",", "."))

    m_gd_forma = re.search(r"(?:cauzione|fideiussione|garanzia fideiussoria)", gd_section, re.IGNORECASE)
    if m_gd_forma:
        gd["forma"] = _clean(m_gd_forma.group(0))

    # Note garanzia definitiva
    m_gd_stip = re.search(r"all'atto\s+della\s+stipul", gd_section or text, re.IGNORECASE)
    m_gd_art = re.search(r"art\.?\s*117[^.]{0,80}", gd_section or text, re.IGNORECASE)
    if m_gd_stip or m_gd_art:
        gd["dovuta"] = True
        parts = []
        if m_gd_stip:
            parts.append("Richiesta all'atto della stipulazione del contratto")
        if m_gd_art:
            parts.append(_clean(m_gd_art.group(0))[:80])
        gd["note"] = ". ".join(parts)

    # ── Polizza RC ───────────────────────────────────────────────────────
    if "polizza" in text_lower and ("responsabilità civile" in text_lower or "rc professionale" in text_lower or "errori e omissioni" in text_lower):
        pol["richiesta"] = True
        m_pol = re.search(
            r"polizza[^.]{0,200}?(?:errori\s+e\s+omissioni|responsabilità\s+civile)[^.]{0,200}",
            text, re.IGNORECASE,
        )
        if m_pol:
            pol["copertura"] = _clean(m_pol.group(0))[:200]

    return gar
