FROM python:3.12-slim

WORKDIR /app

# Dipendenze sistema per pdfplumber
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installa dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il progetto
COPY ai_engine.py server.py pipeline.py field_registry.py pdf_parser.py \
     schemas.py ml_engine.py extract_disciplinari.py \
     config.py database.py utils.py log_config.py \
     index.html ./

# Crea cartelle dati e utente non-root
RUN mkdir -p data/uploads models \
    && adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

# Porta esposta
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Avvia il server
CMD ["python", "server.py"]
