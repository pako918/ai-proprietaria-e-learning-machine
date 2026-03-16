"""Endpoint di estrazione PDF e testo."""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from config import MAX_UPLOAD_SIZE_BYTES, MIN_TEXT_LENGTH
from log_config import get_logger

router = APIRouter(tags=["Extraction"])
logger = get_logger("api.extraction")


def _get_pipeline():
    from pipeline import pipeline
    return pipeline


@router.post("/api/extract")
async def extract_pdf(file: UploadFile = File(...)):
    """Pipeline completa 9 fasi per PDF."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Solo file PDF accettati")
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(400, f"File troppo grande (max {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB)")
    logger.info("Estrazione PDF: %s (%d bytes)", file.filename, len(content))
    result = _get_pipeline().process_pdf(content, file.filename)
    if "error" in result:
        raise HTTPException(422, result["error"])
    return JSONResponse(content=result)


@router.post("/api/extract-text")
async def extract_from_text(payload: dict):
    """Pipeline per testo grezzo."""
    text = payload.get("text", "")
    filename = payload.get("filename", "input.txt")
    if len(text) < MIN_TEXT_LENGTH:
        raise HTTPException(400, "Testo troppo breve")
    result = _get_pipeline().process_text(text, filename)
    return JSONResponse(content=result)
