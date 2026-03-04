"""
AppaltoAI - Backend FastAPI
API REST per estrazione dati PDF, learning e gestione documenti
"""

import io
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from datetime import datetime
from ai_engine import extractor
from pdf_parser import parse_pdf, get_text_with_tables, get_page_for_text, ParsedDocument
from schemas import validate_extraction, full_validation, CoherenceValidator

# ─── Dipendenze: supporta sia FastAPI che fallback HTTP puro ──────────────────
try:
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# PDF TEXT EXTRACTION — usa il nuovo pdf_parser strutturato
# ═════════════════════════════════════════════════════════════════════════════

# Cache ultimo documento parsato per accesso rapido ai metadati
_last_parsed: dict = {}

def extract_text_from_pdf(pdf_bytes: bytes, filename: str = "document.pdf") -> str:
    """Estrae testo da PDF con il nuovo parser strutturato.
    Il ParsedDocument completo è accessibile in _last_parsed[filename]."""
    global _last_parsed
    try:
        parsed = parse_pdf(pdf_bytes, filename)
        _last_parsed[filename] = parsed
        # Usa markdown (testo + tabelle formattate) per la migliore estrazione
        text = get_text_with_tables(parsed)
        if text and len(text.strip()) > 50:
            return text
    except Exception:
        pass

    # Fallback legacy: pdfplumber diretto
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        if text.strip():
            return text
    except Exception:
        pass

    return "[Impossibile estrarre testo dal PDF - potrebbe essere scansionato]"


# ═════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═════════════════════════════════════════════════════════════════════════════

if HAS_FASTAPI:
    app = FastAPI(
        title="AppaltoAI API",
        description="AI per estrazione dati da PDF gare d'appalto",
        version="1.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve frontend statico — index.html è nella root del progetto
    frontend_dir = BASE_DIR
    if (frontend_dir / "index.html").exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir), html=True), name="static")

    # ── ENDPOINTS ─────────────────────────────────────────────────────────

    @app.get("/")
    async def root():
        """Serve la pagina HTML principale"""
        index_file = BASE_DIR / "index.html"
        if index_file.exists():
            from fastapi.responses import HTMLResponse
            return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
        return {"status": "AppaltoAI attivo", "version": "1.0.0"}

    @app.get("/api/health")
    async def health():
        stats = extractor.get_stats()
        from pdf_parser import HAS_FITZ, HAS_PDFPLUMBER
        parser = "fitz+pdfplumber" if HAS_FITZ else ("pdfplumber" if HAS_PDFPLUMBER else "fallback")
        return {
            "status": "ok",
            "pdf_parser": parser,
            "ml_models_loaded": len(extractor.ml_models),
            "stats": stats
        }

    @app.post("/api/extract")
    async def extract_pdf(file: UploadFile = File(...)):
        """Carica PDF ed estrae i dati automaticamente con parsing strutturato"""
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(400, "Solo file PDF accettati")

        content = await file.read()
        if len(content) > 50 * 1024 * 1024:  # 50MB max
            raise HTTPException(400, "File troppo grande (max 50MB)")

        # ── Smart PDF parsing (strutturato) ───────────────────────────
        text = extract_text_from_pdf(content, file.filename)
        if len(text.strip()) < 50:
            raise HTTPException(422, "Impossibile estrarre testo dal PDF. Il file potrebbe essere scansionato senza OCR.")

        # ── AI extraction ─────────────────────────────────────────────
        result = extractor.extract(text, file.filename)
        result["_text_length"] = len(text)
        result["_filename"] = file.filename

        # ── Page sourcing: mappa ogni campo alla pagina sorgente ──────
        parsed = _last_parsed.get(file.filename)
        if parsed and parsed.pages:
            page_sources = {}
            for field, value in result.items():
                if field.startswith('_') or value is None or value == "":
                    continue
                if isinstance(value, (bool, dict, list)):
                    continue
                page = get_page_for_text(parsed, str(value))
                if page:
                    page_sources[field] = page
            result["_page_sources"] = page_sources
            result["_total_pages"] = parsed.total_pages
            result["_is_native_pdf"] = parsed.is_native
            result["_parser_used"] = parsed.parser_used
            result["_pdf_metadata"] = parsed.metadata
            result["_pdf_warnings"] = parsed.warnings
            result["_tables_found"] = len(parsed.tables_json)
            result["_tables_json"] = parsed.tables_json[:20]  # Max 20 tabelle
            result["_chunks_count"] = len(parsed.chunks)

        # ── Coherence validation ──────────────────────────────────────
        try:
            validation = full_validation(result)
            result["_coherence"] = validation["coherence"]
            result["_validation_warnings"] = validation["warnings"]
        except Exception:
            pass

        return JSONResponse(content=result)

    @app.post("/api/extract-text")
    async def extract_from_text(payload: dict):
        """Estrae dati da testo grezzo (per test)"""
        text = payload.get("text", "")
        filename = payload.get("filename", "input.txt")
        if len(text) < 20:
            raise HTTPException(400, "Testo troppo breve")
        result = extractor.extract(text, filename)
        return JSONResponse(content=result)

    @app.post("/api/feedback")
    async def record_feedback(payload: dict):
        """Registra correzione utente per online learning"""
        doc_id = payload.get("doc_id")
        field = payload.get("field")
        original = payload.get("original", "")
        corrected = payload.get("corrected", "")
        snippet = payload.get("text_snippet", "")

        if not all([doc_id, field, corrected]):
            raise HTTPException(400, "Parametri mancanti: doc_id, field, corrected")

        retrained = extractor.record_correction(doc_id, field, original, corrected, snippet)

        return {
            "status": "ok",
            "message": f"Correzione salvata per campo '{field}'",
            "model_retrained": retrained,
            "sample_count": extractor._get_sample_count(field)
        }

    @app.post("/api/train/{field}")
    async def train_model(field: str):
        """Forza riaddestramento modello per un campo"""
        model, message = extractor.train_field_classifier(field)
        return {
            "status": "ok" if model else "error",
            "message": message,
            "field": field
        }

    @app.get("/api/stats")
    async def get_stats():
        return extractor.get_stats()

    @app.get("/api/history")
    async def get_history():
        return extractor.get_history()

    @app.get("/api/history/{doc_id}")
    async def get_document(doc_id: str):
        import sqlite3
        conn = sqlite3.connect(extractor.__class__.__module__)  # workaround
        from ai_engine import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        row = c.execute(
            "SELECT filename, upload_date, extracted_json, corrected_json FROM documents WHERE id=?",
            (doc_id,)
        ).fetchone()
        conn.close()
        if not row:
            raise HTTPException(404, "Documento non trovato")
        return {
            "id": doc_id,
            "filename": row[0],
            "upload_date": row[1],
            "extracted": json.loads(row[2]) if row[2] else {},
            "corrected": json.loads(row[3]) if row[3] else {},
        }

    @app.get("/api/document/{doc_id}/text")
    async def get_doc_text(doc_id: str):
        """Recupera testo completo e snippet di un documento per le correzioni"""
        data = extractor.get_document_text(doc_id)
        if not data:
            raise HTTPException(404, "Documento non trovato")
        return data

    # ── Quarantine & Auto-learn endpoints ─────────────────────────────

    @app.get("/api/quarantine")
    async def get_quarantine():
        """Restituisce i campioni in quarantena in attesa di review."""
        return extractor.get_quarantine()

    @app.post("/api/quarantine/{qid}/approve")
    async def approve_quarantine(qid: int):
        """Approva un campione in quarantena → entra nel training set."""
        result = extractor.approve_quarantine(qid)
        if result.get("status") == "error":
            raise HTTPException(404, result["message"])
        return result

    @app.post("/api/quarantine/{qid}/reject")
    async def reject_quarantine(qid: int):
        """Rifiuta un campione in quarantena → scartato."""
        return extractor.reject_quarantine(qid)

    @app.get("/api/auto-learn/stats")
    async def auto_learn_stats():
        """Statistiche sul safe auto-learning."""
        return extractor.get_auto_learn_stats()

    @app.post("/api/validate")
    async def validate_result(payload: dict):
        """Valida un risultato di estrazione con Pydantic + coherence check."""
        try:
            validation = full_validation(payload)
            return JSONResponse(content=validation)
        except Exception as e:
            raise HTTPException(500, f"Errore validazione: {str(e)}")

    @app.get("/api/pdf-info/{doc_id}")
    async def get_pdf_info(doc_id: str):
        """Ritorna info strutturali del PDF parsato (tabelle, chunks, metadati)."""
        # Cerca nel cache
        for fname, parsed in _last_parsed.items():
            if doc_id in fname or True:  # Serve l'ultimo parsato
                return {
                    "filename": parsed.filename,
                    "is_native": parsed.is_native,
                    "total_pages": parsed.total_pages,
                    "parser_used": parsed.parser_used,
                    "metadata": parsed.metadata,
                    "warnings": parsed.warnings,
                    "tables": parsed.tables_json[:20],
                    "chunks": [
                        {"type": c.chunk_type, "title": c.title,
                         "pages": f"{c.page_start}-{c.page_end}",
                         "section": c.section_path,
                         "length": len(c.content)}
                        for c in parsed.chunks[:50]
                    ],
                }
        raise HTTPException(404, "Documento non in cache")

    if __name__ == "__main__":
        uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

else:
    # ── Fallback: server HTTP puro (stdlib) ───────────────────────────────
    import http.server
    import socketserver
    import urllib.parse
    import cgi

    class AppaltoHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # silenzia log

        def send_json(self, data, status=200):
            body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/api/health":
                self.send_json({"status": "ok", "pdf_parser": "fallback", "ml_models": len(extractor.ml_models)})
            elif path == "/api/stats":
                self.send_json(extractor.get_stats())
            elif path == "/api/history":
                self.send_json(extractor.get_history())
            else:
                # Serve frontend
                frontend_file = BASE_DIR / "index.html"
                if frontend_file.exists():
                    content = frontend_file.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", len(content))
                    self.end_headers()
                    self.wfile.write(content)
                else:
                    self.send_json({"error": "Frontend non trovato"}, 404)

        def do_POST(self):
            path = self.path.split("?")[0]
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            if path == "/api/extract-text":
                try:
                    payload = json.loads(body)
                    result = extractor.extract(payload.get("text", ""), payload.get("filename", ""))
                    self.send_json(result)
                except Exception as e:
                    self.send_json({"error": str(e)}, 500)

            elif path == "/api/feedback":
                try:
                    payload = json.loads(body)
                    retrained = extractor.record_correction(
                        payload["doc_id"], payload["field"],
                        payload.get("original", ""), payload["corrected"],
                        payload.get("text_snippet", "")
                    )
                    self.send_json({"status": "ok", "model_retrained": retrained})
                except Exception as e:
                    self.send_json({"error": str(e)}, 500)
            else:
                self.send_json({"error": "Endpoint non trovato"}, 404)

    def run_fallback_server(port=8000):
        with socketserver.TCPServer(("", port), AppaltoHandler) as httpd:
            print(f"✅ AppaltoAI Server avviato su http://localhost:{port}")
            httpd.serve_forever()

    if __name__ == "__main__":
        run_fallback_server()
