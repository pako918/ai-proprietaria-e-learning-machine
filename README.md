# AppaltoAI — Pipeline 9 Fasi per Gare d'Appalto

Sistema AI **proprietario** per l'estrazione automatica di dati da PDF di gare d'appalto italiane.
Nessuna API esterna. Tutto gira sul tuo server. Training **solo supervisionato**, mai automatico.

---

## 🏗 Architettura

```
appalto-ai/
├── pipeline.py           ← Orchestratore 9 fasi (self-contained)
├── field_registry.py     ← Registro campi (single source of truth)
├── server.py             ← API REST (FastAPI)
├── ai_engine.py          ← Motore legacy (mantenuto per compatibilità)
├── pdf_parser.py         ← Parser PDF (PyMuPDF + pdfplumber)
├── schemas.py            ← Validazione Pydantic + coerenza
├── index.html            ← Frontend completo (4 tab: Analizza, Storico, Training, Admin)
├── data/
│   └── learning.db       ← SQLite (documenti, training samples, model versions)
├── models/               ← Modelli ML versionati (.pkl)
└── Dockerfile            ← Container Docker
```

### Come funziona la Pipeline

```
Fase 1: Upload + Hash Dedup
Fase 2: Parsing deterministico (solo regole regex, NO AI)
Fase 3: NLP specializzato (classificazione + entity recognition)
Fase 4: Costruzione JSON (merge regole + NLP — responsabilità del CODICE)
Fase 5: Validazione automatica (schema, somme, coerenza Pydantic)
Fase 6: Output strutturato + salvataggio
Fase 7: Correzioni utente → dataset annotato proprietario
Fase 8: Retraining controllato (SOLO supervisionato, admin-triggered)
Fase 9: Versionamento modelli (v1, v2, v3 con rollback)
```

**Principi chiave:**
- Il modello NON scrive testo. Riconosce entità, classifica, struttura.
- Il JSON è responsabilità del CODICE, non del modello.
- Mai training automatico. Sempre supervisionato dall'admin.

---

## 🚀 Installazione

### Dipendenze (minime)
```bash
pip install scikit-learn numpy pandas
```

### Con PDF parsing (consigliato)
```bash
pip install pdfplumber scikit-learn numpy pandas
```

### Con server FastAPI (consigliato per produzione)
```bash
pip install fastapi uvicorn python-multipart pdfplumber scikit-learn numpy pandas
```

---

## ▶️ Avvio

```bash
chmod +x start.sh
./start.sh
```

Oppure direttamente:
```bash
python3 backend/server.py
```

Apri il browser su: **http://localhost:8000**

---

## 📡 API REST

### Estrai dati da PDF
```bash
curl -X POST http://localhost:8000/api/extract \
  -F "file=@bando_gara.pdf"
```

### Estrai da testo
```bash
curl -X POST http://localhost:8000/api/extract-text \
  -H "Content-Type: application/json" \
  -d '{"text": "...testo del bando...", "filename": "bando.txt"}'
```

**Risposta JSON:**
```json
{
  "stazione_appaltante": "Comune di Milano",
  "oggetto_gara": "Servizi di pulizia edifici comunali",
  "cig": "ABC1234567",
  "importo_base": "150.000,00",
  "criterio_aggiudicazione": "Offerta Economicamente Più Vantaggiosa (OEPV)",
  "punteggio_qualita": 70,
  "punteggio_prezzo": 30,
  "scadenza_offerte": "30/06/2025 12:00",
  "tipo_procedura": "Procedura Aperta",
  "categorie_soa": ["OG1", "OS6"],
  "_confidence": 78.5,
  "_doc_id": "a1b2c3d4e5f6",
  "_extraction_method": "rules"
}
```

### Registra correzione (attiva learning)
```bash
curl -X POST http://localhost:8000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "a1b2c3d4e5f6",
    "field": "stazione_appaltante",
    "original": "Comune",
    "corrected": "Comune di Milano",
    "text_snippet": "...porzione di testo del PDF..."
  }'
```

### Forza riadddestramento modello
```bash
curl -X POST http://localhost:8000/api/train/stazione_appaltante
```

### Statistiche AI
```bash
curl http://localhost:8000/api/stats
```

### Storico documenti
```bash
curl http://localhost:8000/api/history
```

---

## 🔗 Integrazione con il tuo sito

### Opzione A — iFrame
```html
<iframe src="http://localhost:8000" width="100%" height="800px" frameborder="0"></iframe>
```

### Opzione B — chiamate API dal tuo JS
```javascript
// Upload PDF dal tuo sito
async function analizzaGara(pdfFile) {
  const formData = new FormData();
  formData.append('file', pdfFile);
  
  const res = await fetch('http://TUO_SERVER:8000/api/extract', {
    method: 'POST',
    body: formData
  });
  
  const dati = await res.json();
  
  // Popola i campi del tuo sito
  document.getElementById('importo').value = dati.importo_base;
  document.getElementById('oggetto').value = dati.oggetto_gara;
  document.getElementById('cig').value = dati.cig;
  // ... etc
  
  return dati;
}
```

### Opzione C — Reverse proxy con Nginx
```nginx
location /appalto-ai/ {
    proxy_pass http://127.0.0.1:8000/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

---

## 🧠 Campi estratti automaticamente

| Campo | Descrizione |
|-------|-------------|
| `stazione_appaltante` | Ente committente |
| `oggetto_gara` | Descrizione appalto |
| `cig` | Codice Identificativo Gara |
| `cup` | Codice Unico Progetto |
| `cpv` | Codice CPV |
| `importo_base` | Importo a base di gara |
| `oneri_sicurezza` | Oneri della sicurezza |
| `criterio_aggiudicazione` | OEPV o Massimo Ribasso |
| `punteggio_qualita` | Punti qualità (OEPV) |
| `punteggio_prezzo` | Punti prezzo (OEPV) |
| `scadenza_offerte` | Termine presentazione offerte |
| `tipo_procedura` | Aperta/Ristretta/Negoziata/etc |
| `durata_contratto` | Durata in mesi/anni |
| `categorie_soa` | Categorie SOA (OG, OS) |
| `classifica_soa` | Classifica SOA |
| `cauzione_provvisoria` | Importo cauzione |
| `responsabile_procedimento` | RUP |
| `subappalto` | Condizioni subappalto |
| `soglia_anomalia` | Soglia anomalia offerte |
| `sopralluogo_obbligatorio` | true/false |
| `requisiti_tecnici` | Lista requisiti |

---

## 📈 Come migliora nel tempo

1. **Carica un PDF** → l'AI estrae i dati con le regole
2. **Correggi i campi errati** → clicca ✏️ su ogni card
3. **Ogni correzione viene salvata** nel database SQLite
4. **Dopo 10 correzioni per campo**, il modello ML si riaddestrà automaticamente
5. **I PDF successivi** vengono analizzati con regole + ML = maggiore precisione

---

## ⚙️ Configurazione avanzata

### Aggiungere pattern personalizzati
Modifica `backend/ai_engine.py`, sezione `PATTERNS`:
```python
PATTERNS = {
    "mio_campo_custom": [
        r"(?:mia etichetta)[:\s]+([^\n]{5,100})",
    ],
    # ...
}
```

### Cambiare porta
```bash
python3 backend/server.py  # modifica port=8000 in server.py
```

### Backup database
```bash
cp data/learning.db data/learning_backup_$(date +%Y%m%d).db
```

---

## 📋 Requisiti di sistema

- Python 3.8+
- 256MB RAM minimo
- Qualsiasi OS (Linux, Windows, macOS)
- Nessuna connessione internet richiesta
