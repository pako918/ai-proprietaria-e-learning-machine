"""
extractors – pacchetto modulare di estrazione dati da disciplinari di gara.

Re-esporta le API pubbliche in modo che il codice esistente possa fare::

    from extractors import extract_rules_based, flatten_for_pipeline
    from extractors import extract_from_pdf_bytes, extract_from_text_direct
    from extractors import extract_disciplinare, extract_all_disciplinari
    from extractors import extract_text_from_pdf
"""

from .main import (
    extract_rules_based,
    extract_from_pdf_bytes,
    extract_from_text_direct,
    extract_disciplinare,
    extract_all_disciplinari,
)
from .flatten import flatten_for_pipeline
from .pdf import extract_text_from_pdf

__all__ = [
    "extract_rules_based",
    "flatten_for_pipeline",
    "extract_from_pdf_bytes",
    "extract_from_text_direct",
    "extract_disciplinare",
    "extract_all_disciplinari",
    "extract_text_from_pdf",
]
