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
COPY *.py ./
COPY index.html ./
COPY routers/ ./routers/
COPY extractors/ ./extractors/
COPY doe/ ./doe/

# Copia e registra l'entrypoint per gestire permessi sui volumi
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Crea cartelle dati e utente non-root
RUN mkdir -p data/uploads models \
    && adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app

# Porta esposta
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Entry point: sistema permessi sui volumi e poi avvia il processo come `appuser`
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "server.py"]
