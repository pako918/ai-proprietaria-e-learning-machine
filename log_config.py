"""
AppaltoAI — Logging Configuration
Setup centralizzato del logging per tutti i moduli.
"""

import logging
import sys
from config import LOG_LEVEL


def setup_logging():
    """Configura il logging per l'intera applicazione."""
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Riduci il rumore dei logger di terze parti
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sklearn").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Ritorna un logger con namespace appaltoai."""
    return logging.getLogger(f"appaltoai.{name}")
