"""
AppaltoAI - Backend FastAPI
API REST — Pipeline 9 fasi — Training SOLO supervisionato
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    BASE_DIR, CORS_ORIGINS, SERVER_HOST, SERVER_PORT,
    API_VERSION, APP_TITLE, MAX_UPLOAD_SIZE_BYTES, MIN_TEXT_LENGTH,
)
from log_config import setup_logging, get_logger
from database import get_connection
from pipeline import pipeline
from field_registry import registry
from schemas import full_validation

setup_logging()
logger = get_logger("server")

# ── Dipendenze: FastAPI o fallback HTTP ───────────────────────────────────────
try:
    from fastapi import FastAPI, UploadFile, File, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


# ═════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═════════════════════════════════════════════════════════════════════════════

if HAS_FASTAPI:
    app = FastAPI(
        title=APP_TITLE,
        description="AI proprietaria per estrazione dati da bandi di gara — Pipeline 9 fasi",
        version=API_VERSION,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    if (BASE_DIR / "index.html").exists():
        app.mount("/static", StaticFiles(directory=str(BASE_DIR), html=True), name="static")

    # ══════════════════════════════════════════════════════════════════════
    # CORE ENDPOINTS
    # ══════════════════════════════════════════════════════════════════════

    @app.get("/")
    async def root():
        index_file = BASE_DIR / "index.html"
        if index_file.exists():
            return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
        return {"status": "AppaltoAI attivo", "version": API_VERSION}

    @app.get("/api/health")
    async def health():
        stats = pipeline.get_stats()
        return {
            "status": "ok",
            "version": API_VERSION,
            "architecture": "ML-powered extraction pipeline",
            "learning_mode": "data-driven supervised",
            "ml_engine": pipeline.get_ml_status(),
            "stats": stats,
        }

    # ══════════════════════════════════════════════════════════════════════
    # FASI 1-6: ESTRAZIONE
    # ══════════════════════════════════════════════════════════════════════

    @app.post("/api/extract")
    async def extract_pdf(file: UploadFile = File(...)):
        """Pipeline completa 9 fasi per PDF."""
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(400, "Solo file PDF accettati")
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(400, f"File troppo grande (max {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB)")
        logger.info("Estrazione PDF: %s (%d bytes)", file.filename, len(content))
        result = pipeline.process_pdf(content, file.filename)
        if "error" in result:
            raise HTTPException(422, result["error"])
        return JSONResponse(content=result)

    @app.post("/api/extract-text")
    async def extract_from_text(payload: dict):
        """Pipeline per testo grezzo."""
        text = payload.get("text", "")
        filename = payload.get("filename", "input.txt")
        if len(text) < MIN_TEXT_LENGTH:
            raise HTTPException(400, "Testo troppo breve")
        result = pipeline.process_text(text, filename)
        return JSONResponse(content=result)

    # ══════════════════════════════════════════════════════════════════════
    # FASE 7: CORREZIONI → DATASET PROPRIETARIO
    # ══════════════════════════════════════════════════════════════════════

    @app.post("/api/feedback")
    async def record_feedback(payload: dict):
        """Registra correzione utente → training sample (NO auto-training)."""
        doc_id = payload.get("doc_id")
        field = payload.get("field")
        original = payload.get("original", "")
        corrected = payload.get("corrected", "")
        snippet = payload.get("text_snippet", "")
        if not all([doc_id, field, corrected]):
            raise HTTPException(400, "Parametri mancanti: doc_id, field, corrected")
        result = pipeline.record_correction(doc_id, field, original, corrected, snippet)
        return result

    # ── Corrections CRUD ──────────────────────────────────────────────────

    @app.get("/api/corrections")
    async def get_corrections():
        return pipeline.get_corrections()

    @app.get("/api/corrections/stats")
    async def get_corrections_stats():
        return pipeline.get_corrections_stats()

    @app.put("/api/corrections/{correction_id}")
    async def update_correction(correction_id: int, payload: dict):
        sample_id = payload.get("sample_id")
        result = pipeline.update_correction(
            correction_id=correction_id, sample_id=sample_id, data=payload
        )
        if result.get("status") == "error":
            raise HTTPException(400, result["message"])
        return result

    @app.delete("/api/corrections/{correction_id}")
    async def delete_correction(correction_id: int):
        return pipeline.delete_correction(correction_id=correction_id)

    @app.delete("/api/training-sample/{sample_id}")
    async def delete_training_sample(sample_id: int):
        return pipeline.delete_correction(sample_id=sample_id)

    # ══════════════════════════════════════════════════════════════════════
    # FASE 8: RETRAINING SUPERVISIONATO (MAI AUTOMATICO)
    # ══════════════════════════════════════════════════════════════════════

    @app.post("/api/train/{field}")
    async def train_model(field: str):
        """Training supervisionato — attivato SOLO dall'admin."""
        result = pipeline.train_field(field)
        return result

    @app.post("/api/models/{field}/train")
    async def supervised_training(field: str):
        """Alias: training supervisionato."""
        return await train_model(field)

    # ══════════════════════════════════════════════════════════════════════
    # FASE 9: MODEL VERSIONING + ROLLBACK
    # ══════════════════════════════════════════════════════════════════════

    @app.get("/api/models/versions")
    async def model_versions(field: str = Query(None)):
        return pipeline.get_model_versions(field)

    @app.post("/api/models/{field}/rollback")
    async def model_rollback(field: str):
        result = pipeline.rollback_model(field)
        if result.get("status") == "error":
            raise HTTPException(400, result["message"])
        return result

    # ══════════════════════════════════════════════════════════════════════
    # STATS, HISTORY, DOCUMENT
    # ══════════════════════════════════════════════════════════════════════

    @app.get("/api/stats")
    async def get_stats():
        return pipeline.get_stats()

    @app.get("/api/history")
    async def get_history():
        return pipeline.get_history()

    @app.get("/api/history/{doc_id}")
    async def get_document(doc_id: str):
        with get_connection(readonly=True) as conn:
            row = conn.execute(
                "SELECT filename, upload_date, extracted_json, corrected_json "
                "FROM documents WHERE id=?",
                (doc_id,)
            ).fetchone()
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
        data = pipeline.get_document_text(doc_id)
        if not data:
            raise HTTPException(404, "Documento non trovato")
        return data

    @app.get("/api/pdf-info/{doc_id}")
    async def get_pdf_info(doc_id: str):
        """Info strutturali PDF (tabelle, chunks, metadati)."""
        for fname, parsed in pipeline._last_parsed.items():
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

    # ══════════════════════════════════════════════════════════════════════
    # VALIDAZIONE
    # ══════════════════════════════════════════════════════════════════════

    @app.post("/api/validate")
    async def validate_result(payload: dict):
        try:
            validation = full_validation(payload)
            return JSONResponse(content=validation)
        except Exception as e:
            raise HTTPException(500, f"Errore validazione: {str(e)}")

    # ══════════════════════════════════════════════════════════════════════
    # FIELD REGISTRY — Campi custom
    # ══════════════════════════════════════════════════════════════════════

    @app.get("/api/fields")
    async def get_fields():
        return {
            "sections": registry.to_sections_json(),
            "all_keys": registry.get_keys(),
        }

    @app.get("/api/fields/custom")
    async def get_custom_fields():
        return [f.to_dict() for f in registry.get_custom_fields()]

    @app.post("/api/fields/custom")
    async def add_custom_field(payload: dict):
        try:
            fd = registry.add_custom_field(payload)
            return {"status": "ok", "field": fd.to_dict()}
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.put("/api/fields/custom/{key}")
    async def update_custom_field(key: str, payload: dict):
        try:
            fd = registry.update_custom_field(key, payload)
            return {"status": "ok", "field": fd.to_dict()}
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.delete("/api/fields/custom/{key}")
    async def delete_custom_field(key: str):
        ok = registry.delete_custom_field(key)
        if not ok:
            raise HTTPException(404, f"Campo custom '{key}' non trovato")
        return {"status": "ok"}

    # ══════════════════════════════════════════════════════════════════════    # ML ENGINE — Machine Learning (il modello impara dai dati)
    # ══════════════════════════════════════════════════════════════════

    @app.get("/api/ml/status")
    async def ml_status():
        """Stato del motore ML: modelli attivi, dati, configurazione."""
        return pipeline.get_ml_status()

    @app.get("/api/ml/quality")
    async def ml_quality():
        """Report qualità dati e modelli con raccomandazioni."""
        return pipeline.get_ml_quality()

    @app.post("/api/ml/train")
    async def ml_train_all():
        """Addestra tutti i modelli con dati sufficienti."""
        return pipeline.train_all()

    @app.post("/api/ml/train/{field}")
    async def ml_train_field(field: str):
        """Addestra il modello per un campo specifico."""
        return pipeline.train_field(field)

    @app.get("/api/ml/learning-curve/{field}")
    async def ml_learning_curve(field: str):
        """Curva di apprendimento: come l'accuracy migliora con più dati."""
        from ml_engine import ml_engine as ml_eng
        return ml_eng.get_learning_curve(field)

    @app.get("/api/ml/data")
    async def ml_data_stats():
        """Statistiche dataset di training ML."""
        from ml_engine import ml_engine as ml_eng
        return ml_eng.data.get_data_quality()

    @app.get("/api/ml/data/{field}")
    async def ml_data_field(field: str):
        """Statistiche dati per un campo specifico."""
        from ml_engine import ml_engine as ml_eng
        return ml_eng.data.get_data_quality(field)

    @app.post("/api/ml/rollback/{field}")
    async def ml_rollback(field: str):
        """Ripristina il modello precedente."""
        from ml_engine import ml_engine as ml_eng
        result = ml_eng.rollback_field(field)
        if result.get("status") == "error":
            raise HTTPException(400, result["message"])
        return result

    @app.get("/api/ml/versions")
    async def ml_model_versions(field: str = Query(None)):
        """Storico versioni modelli ML."""
        from ml_engine import ml_engine as ml_eng
        return ml_eng.get_model_versions(field)

    # ══════════════════════════════════════════════════════════════════    # PIPELINE STATUS
    # ══════════════════════════════════════════════════════════════════════

    @app.get("/api/pipeline/status")
    async def pipeline_status():
        stats = pipeline.get_stats()
        versions = pipeline.get_model_versions()
        ml_status = pipeline.get_ml_status()
        return {
            "pipeline_version": API_VERSION,
            "architecture": "ML-powered extraction",
            "learning_mode": "data-driven supervised",
            "ml_engine": ml_status,
            "stats": stats,
            "models": versions,
        }

    # ── LEGACY COMPATIBILITY STUBS ────────────────────────────────────────
    # Endpoint vecchi restituiscono risposte vuote compatibili

    @app.get("/api/quarantine")
    async def get_quarantine():
        return []

    @app.get("/api/auto-learn/stats")
    async def auto_learn_stats():
        return {"message": "Auto-learning disabilitato. Usa training supervisionato.", "auto_learn": False}

    @app.post("/api/quarantine/{qid}/approve")
    async def approve_quarantine(qid: int):
        return {"status": "deprecated", "message": "Quarantine rimossa. Usa correzioni manuali."}

    @app.post("/api/quarantine/{qid}/reject")
    async def reject_quarantine(qid: int):
        return {"status": "deprecated", "message": "Quarantine rimossa."}

    if __name__ == "__main__":
        logger.info("Avvio AppaltoAI su %s:%d", SERVER_HOST, SERVER_PORT)
        uvicorn.run("server:app", host=SERVER_HOST, port=SERVER_PORT, reload=True)

else:
    # ── Fallback: server HTTP (stdlib) ────────────────────────────────────
    import http.server
    import socketserver

    class AppaltoHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

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
            self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/api/health":
                self.send_json({"status": "ok", "version": "2.0.0", "auto_learn": False})
            elif path == "/api/stats":
                self.send_json(pipeline.get_stats())
            elif path == "/api/history":
                self.send_json(pipeline.get_history())
            elif path == "/api/corrections":
                self.send_json(pipeline.get_corrections())
            elif path == "/api/corrections/stats":
                self.send_json(pipeline.get_corrections_stats())
            elif path == "/api/models/versions":
                self.send_json(pipeline.get_model_versions())
            elif path == "/api/fields":
                self.send_json({"sections": registry.to_sections_json(), "all_keys": registry.get_keys()})
            else:
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
                    result = pipeline.process_text(payload.get("text", ""), payload.get("filename", "input.txt"))
                    self.send_json(result)
                except Exception as e:
                    self.send_json({"error": str(e)}, 500)
            elif path == "/api/feedback":
                try:
                    payload = json.loads(body)
                    result = pipeline.record_correction(
                        payload["doc_id"], payload["field"],
                        payload.get("original", ""), payload["corrected"],
                        payload.get("text_snippet", "")
                    )
                    self.send_json(result)
                except Exception as e:
                    self.send_json({"error": str(e)}, 500)
            elif path.startswith("/api/train/"):
                field = path.split("/")[-1]
                self.send_json(pipeline.train_field(field))
            else:
                self.send_json({"error": "Endpoint non trovato"}, 404)

        def do_PUT(self):
            self.send_json({"error": "PUT non supportato nel fallback"}, 501)

        def do_DELETE(self):
            self.send_json({"error": "DELETE non supportato nel fallback"}, 501)

    def run_fallback_server(port=8000):
        with socketserver.TCPServer(("", port), AppaltoHandler) as httpd:
            print(f"✅ AppaltoAI Server v2.0 su http://localhost:{port}")
            httpd.serve_forever()

    if __name__ == "__main__":
        run_fallback_server()
