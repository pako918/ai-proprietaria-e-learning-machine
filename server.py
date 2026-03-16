"""
AppaltoAI — Backend FastAPI
Entry point dell'applicazione. Crea l'app e registra i router modulari.
Il fallback HTTP (stdlib) è disponibile se FastAPI non è installato.
"""

import json

from config import (
    BASE_DIR, CORS_ORIGINS, SERVER_HOST, SERVER_PORT,
    API_VERSION, APP_TITLE,
)
from log_config import setup_logging, get_logger

setup_logging()
logger = get_logger("server")


# ── Dipendenze: FastAPI o fallback HTTP ───────────────────────────────────────
try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


def create_app() -> "FastAPI":
    """App factory: crea e configura l'applicazione FastAPI."""
    application = FastAPI(
        title=APP_TITLE,
        description="AI proprietaria per estrazione dati da bandi di gara — Pipeline 9 fasi",
        version=API_VERSION,
    )

    # CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    # Static files
    if (BASE_DIR / "index.html").exists():
        application.mount("/static", StaticFiles(directory=str(BASE_DIR), html=True), name="static")

    # Registra i router modulari
    from routers.core import router as core_router
    from routers.extraction import router as extraction_router
    from routers.corrections import router as corrections_router
    from routers.training import router as training_router
    from routers.ml_routes import router as ml_router
    from routers.fields import router as fields_router
    from routers.documents import router as documents_router
    from routers.learning import router as learning_router
    from routers.adaptive import router as adaptive_router
    from routers.doe import router as doe_router

    application.include_router(core_router)
    application.include_router(extraction_router)
    application.include_router(corrections_router)
    application.include_router(training_router)
    application.include_router(ml_router)
    application.include_router(fields_router)
    application.include_router(documents_router)
    application.include_router(learning_router)
    application.include_router(adaptive_router)
    application.include_router(doe_router)

    return application


# ═════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═════════════════════════════════════════════════════════════════════════════

if HAS_FASTAPI:
    app = create_app()

    if __name__ == "__main__":
        logger.info("Avvio AppaltoAI su %s:%d", SERVER_HOST, SERVER_PORT)
        uvicorn.run("server:app", host=SERVER_HOST, port=SERVER_PORT, reload=True)

else:
    # ── Fallback: server HTTP (stdlib) ────────────────────────────────────
    import http.server
    import socketserver

    from pipeline import pipeline
    from field_registry import registry
    from database import get_connection

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
                self.send_json({"status": "ok", "version": API_VERSION, "auto_learn": False})
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
            print(f"✅ AppaltoAI Server v{API_VERSION} su http://localhost:{port}")
            httpd.serve_forever()

    if __name__ == "__main__":
        run_fallback_server()
