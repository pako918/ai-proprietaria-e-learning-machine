"""
AppaltoAI — Smart PDF Parser
Parsing intelligente multi-strategia:
  1. Nativo vs Scansionato detection
  2. PyMuPDF (fitz) per estrazione strutturata con metadati pagina
  3. Tabelle → Markdown/JSON strutturato
  4. Chunking semantico per struttura legale (Articoli, Commi, Sezioni)
  5. Page sourcing per ogni segmento
"""

import re
import io
from dataclasses import dataclass, field
from typing import Optional

# ── Parser backends ───────────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


# ═════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PageContent:
    """Contenuto di una singola pagina con metadati."""
    page_num: int
    text: str
    tables: list = field(default_factory=list)       # Lista di tabelle [{headers:[], rows:[[]]}]
    markdown: str = ""
    has_images: bool = False
    word_count: int = 0


@dataclass
class LegalChunk:
    """Segmento semantico di un documento legale."""
    chunk_type: str       # "articolo", "comma", "sezione", "tabella", "header", "body"
    title: str            # Es: "Art. 5 — Criteri di aggiudicazione"
    content: str          # Testo del chunk
    markdown: str         # Versione Markdown con formattazione
    page_start: int       # Pagina di inizio
    page_end: int         # Pagina di fine
    section_path: str     # Percorso gerarchico: "Sezione II > Art. 5 > Comma 3"
    tables: list = field(default_factory=list)


@dataclass
class ParsedDocument:
    """Documento PDF processato con tutti i livelli di struttura."""
    filename: str
    is_native: bool                # True = testo nativo, False = scansionato/immagine
    total_pages: int
    full_text: str                 # Testo completo plain
    full_markdown: str             # Markdown completo con tabelle e intestazioni
    pages: list                    # Lista di PageContent
    chunks: list                   # Lista di LegalChunk (chunking semantico)
    tables_json: list              # Tutte le tabelle in formato JSON strutturato
    metadata: dict = field(default_factory=dict)  # Metadati PDF (autore, creazione, etc.)
    parser_used: str = ""
    warnings: list = field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════════════
# NATIVE VS SCANNED DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def detect_pdf_type(pdf_bytes: bytes) -> dict:
    """Rileva se il PDF è nativo (testo) o scansionato (immagine).
    Ritorna: {'is_native': bool, 'confidence': float, 'text_ratio': float, 'has_images': bool}"""
    result = {"is_native": True, "confidence": 0.5, "text_ratio": 0.0, "has_images": False, "pages": 0}

    if HAS_FITZ:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            result["pages"] = len(doc)
            total_text_blocks = 0
            total_image_blocks = 0
            total_chars = 0

            for page in doc:
                blocks = page.get_text("dict")["blocks"]
                for b in blocks:
                    if b.get("type") == 0:  # text block
                        total_text_blocks += 1
                        for line in b.get("lines", []):
                            for span in line.get("spans", []):
                                total_chars += len(span.get("text", ""))
                    elif b.get("type") == 1:  # image block
                        total_image_blocks += 1
                        result["has_images"] = True

            doc.close()

            total_blocks = total_text_blocks + total_image_blocks
            if total_blocks > 0:
                result["text_ratio"] = total_text_blocks / total_blocks
            else:
                result["text_ratio"] = 0.0

            # Decisione:
            # - Molti blocchi testo + molti caratteri → nativo
            # - Pochi blocchi testo + immagini → scansionato
            chars_per_page = total_chars / max(result["pages"], 1)
            if chars_per_page > 200 and result["text_ratio"] > 0.3:
                result["is_native"] = True
                result["confidence"] = min(0.99, 0.5 + result["text_ratio"] * 0.5)
            elif chars_per_page < 50 and total_image_blocks > 0:
                result["is_native"] = False
                result["confidence"] = 0.85
            else:
                result["is_native"] = chars_per_page > 100
                result["confidence"] = 0.6

        except Exception:
            pass

    return result


# ═════════════════════════════════════════════════════════════════════════════
# PyMuPDF STRUCTURED EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def _extract_with_fitz(pdf_bytes: bytes, filename: str) -> Optional[ParsedDocument]:
    """Estrazione strutturata con PyMuPDF: testo + tabelle + metadati per pagina."""
    if not HAS_FITZ:
        return None

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return None

    pages = []
    all_tables_json = []
    full_text_parts = []
    full_md_parts = []

    # Metadati PDF
    meta = doc.metadata or {}
    metadata = {
        k: v for k, v in {
            "autore": meta.get("author", ""),
            "creatore": meta.get("creator", ""),
            "titolo": meta.get("title", ""),
            "soggetto": meta.get("subject", ""),
            "produttore": meta.get("producer", ""),
            "data_creazione": meta.get("creationDate", ""),
        }.items() if v
    }

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1

        # ── Testo strutturato per blocchi ─────────────────────────────
        blocks = page.get_text("dict")["blocks"]
        page_text_parts = []
        page_md_parts = []
        has_images = False

        for block in blocks:
            if block.get("type") == 1:
                has_images = True
                continue
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                line_text = ""
                line_md = ""
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text.strip():
                        continue
                    size = span.get("size", 10)
                    flags = span.get("flags", 0)
                    is_bold = bool(flags & 2**4)  # bit 4 = bold

                    # Formattazione Markdown in base a font size e bold
                    if is_bold and size >= 14:
                        line_md += f"## {text} "
                    elif is_bold and size >= 12:
                        line_md += f"### {text} "
                    elif is_bold:
                        line_md += f"**{text}** "
                    else:
                        line_md += text + " "

                    line_text += text + " "

                line_text = line_text.strip()
                line_md = line_md.strip()
                if line_text:
                    page_text_parts.append(line_text)
                    page_md_parts.append(line_md)

        page_text = "\n".join(page_text_parts)
        page_md = "\n".join(page_md_parts)

        # ── Tabelle (estrae con find_tables di fitz) ─────────────────
        page_tables = []
        try:
            tabs = page.find_tables()
            for tab in tabs:
                table_data = tab.extract()
                if not table_data or len(table_data) < 2:
                    continue

                # Prima riga = headers, resto = dati
                headers = [str(h).strip() if h else "" for h in table_data[0]]
                rows = []
                for row in table_data[1:]:
                    rows.append([str(c).strip() if c else "" for c in row])

                table_json = {"headers": headers, "rows": rows, "page": page_num}
                page_tables.append(table_json)
                all_tables_json.append(table_json)

                # Converti tabella in Markdown
                md_table = _table_to_markdown(headers, rows)
                page_md += f"\n\n{md_table}\n"

        except Exception:
            pass

        # ── Fallback tabelle con pdfplumber se fitz non ha trovato nulla ──
        if not page_tables and HAS_PDFPLUMBER:
            try:
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as plumber_pdf:
                    if page_idx < len(plumber_pdf.pages):
                        plumber_page = plumber_pdf.pages[page_idx]
                        plumber_tables = plumber_page.extract_tables() or []
                        for pt in plumber_tables:
                            if not pt or len(pt) < 2:
                                continue
                            headers = [str(h).strip() if h else "" for h in pt[0]]
                            rows = [[str(c).strip() if c else "" for c in row] for row in pt[1:]]
                            table_json = {"headers": headers, "rows": rows, "page": page_num}
                            page_tables.append(table_json)
                            all_tables_json.append(table_json)
                            page_md += f"\n\n{_table_to_markdown(headers, rows)}\n"
            except Exception:
                pass

        page_content = PageContent(
            page_num=page_num,
            text=page_text,
            tables=page_tables,
            markdown=page_md,
            has_images=has_images,
            word_count=len(page_text.split())
        )
        pages.append(page_content)
        full_text_parts.append(f"[Pagina {page_num}]\n{page_text}")
        full_md_parts.append(f"\n---\n*Pagina {page_num}*\n\n{page_md}")

    doc.close()

    full_text = "\n\n".join(full_text_parts)
    full_markdown = "\n".join(full_md_parts)

    # Chunking semantico
    chunks = _semantic_chunk(full_text, full_markdown, pages)

    # Warnings
    warnings = []
    scanned_pages = [p.page_num for p in pages if p.word_count < 30 and p.has_images]
    if scanned_pages:
        warnings.append(f"Pagine probabilmente scansionate (poco testo): {scanned_pages}")
    if not all_tables_json:
        warnings.append("Nessuna tabella strutturata rilevata nel PDF")

    return ParsedDocument(
        filename=filename,
        is_native=detect_pdf_type(pdf_bytes)["is_native"],
        total_pages=len(pages),
        full_text=full_text,
        full_markdown=full_markdown,
        pages=pages,
        chunks=chunks,
        tables_json=all_tables_json,
        metadata=metadata,
        parser_used="fitz+pdfplumber",
        warnings=warnings,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PDFPLUMBER FALLBACK
# ═════════════════════════════════════════════════════════════════════════════

def _extract_with_pdfplumber(pdf_bytes: bytes, filename: str) -> Optional[ParsedDocument]:
    """Fallback con pdfplumber quando PyMuPDF non è disponibile."""
    if not HAS_PDFPLUMBER:
        return None

    try:
        pdf = pdfplumber.open(io.BytesIO(pdf_bytes))
    except Exception:
        return None

    pages = []
    all_tables_json = []
    full_text_parts = []
    full_md_parts = []

    for page_idx, page in enumerate(pdf.pages):
        page_num = page_idx + 1
        page_text = page.extract_text() or ""

        # Tabelle
        page_tables = []
        for pt in (page.extract_tables() or []):
            if not pt or len(pt) < 2:
                continue
            headers = [str(h).strip() if h else "" for h in pt[0]]
            rows = [[str(c).strip() if c else "" for c in row] for row in pt[1:]]
            table_json = {"headers": headers, "rows": rows, "page": page_num}
            page_tables.append(table_json)
            all_tables_json.append(table_json)

        # Markdown semplice (senza info font)
        page_md = page_text
        for tbl in page_tables:
            page_md += f"\n\n{_table_to_markdown(tbl['headers'], tbl['rows'])}\n"

        page_content = PageContent(
            page_num=page_num,
            text=page_text,
            tables=page_tables,
            markdown=page_md,
            has_images=bool(page.images),
            word_count=len(page_text.split())
        )
        pages.append(page_content)
        full_text_parts.append(f"[Pagina {page_num}]\n{page_text}")
        full_md_parts.append(f"\n---\n*Pagina {page_num}*\n\n{page_md}")

    pdf.close()

    full_text = "\n\n".join(full_text_parts)
    full_markdown = "\n".join(full_md_parts)
    chunks = _semantic_chunk(full_text, full_markdown, pages)

    return ParsedDocument(
        filename=filename,
        is_native=True,
        total_pages=len(pages),
        full_text=full_text,
        full_markdown=full_markdown,
        pages=pages,
        chunks=chunks,
        tables_json=all_tables_json,
        metadata={},
        parser_used="pdfplumber",
        warnings=[],
    )


# ═════════════════════════════════════════════════════════════════════════════
# TABLE → MARKDOWN CONVERSION
# ═════════════════════════════════════════════════════════════════════════════

def _table_to_markdown(headers: list, rows: list) -> str:
    """Converte una tabella in formato Markdown."""
    if not headers:
        return ""
    # Calcola larghezze
    widths = [max(len(h), 3) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    # Header
    hdr = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"

    # Rows
    row_lines = []
    for row in rows:
        cells = []
        for i in range(len(headers)):
            val = row[i] if i < len(row) else ""
            cells.append(val.ljust(widths[i]))
        row_lines.append("| " + " | ".join(cells) + " |")

    return "\n".join([hdr, sep] + row_lines)


# ═════════════════════════════════════════════════════════════════════════════
# SEMANTIC CHUNKING — Struttura legale italiana
# ═════════════════════════════════════════════════════════════════════════════

# Patterns per struttura legale
_SECTION_PATTERNS = [
    # Sezioni, Titoli, Capi
    (r'^(?:SEZIONE|PARTE|TITOLO|CAPO)\s+[IVX\d]+[.:\s\-–—]+(.{5,120})', "sezione"),
    # Articoli
    (r'^(?:Art(?:icolo)?\.?\s*(\d+(?:[\.\-]\w+)?))\s*[.:\s\-–—]+(.{5,200})', "articolo"),
    (r'^(\d{1,3})\s*[.)]\s+([A-Z][^\n]{5,150})', "articolo"),
    # Commi e lettere
    (r'^(?:(\d+)\.\s+)', "comma"),
    (r'^(?:([a-z])\)\s+)', "lettera"),
]

# Sezioni chiave dei disciplinari
_KEY_SECTIONS = [
    r'requisiti\s+di\s+partecipazione',
    r'criteri?\s+(?:di\s+)?(?:valutazione|aggiudicazione|premianti)',
    r'offerta\s+tecnic[ao]',
    r'offerta\s+economic[ao]',
    r'modalita\s+(?:di\s+)?presentazione\s+(?:dell[ae]\s+)?offert[ae]',
    r'garanzi[ae]\s+(?:provvisori[ae]|definitiv[ae])',
    r'subappalto',
    r'avvalimento',
    r'cause\s+(?:di\s+)?esclusione',
    r'sopralluogo',
    r'soccorso\s+istruttorio',
    r'anomalia\s+(?:dell[ae]\s+)?offert[ae]',
    r'durata\s+(?:del(?:lo)?\s+)?(?:contratto|accordo|servizio)',
    r'importo?\s+(?:a\s+)?base\s+(?:di\s+)?(?:gara|asta)',
    r'punteggi[oi]?\s+(?:massim[oi]|tecnic[oi]|economic[oi])',
    r'soggetti?\s+ammess[ie]',
    r'termine.*offert[ae]',
    r'procedura\s+(?:di\s+)?gara',
]


def _semantic_chunk(full_text: str, full_markdown: str, pages: list) -> list:
    """Suddivide il documento in chunk semantici basati sulla struttura legale.
    Identifica Articoli, Commi, Sezioni e tabelle."""
    chunks = []
    lines = full_text.split("\n")
    current_section = ""
    current_article = ""
    current_chunk_lines = []
    current_chunk_start_page = 1
    current_chunk_type = "body"
    current_title = ""

    # Mappa linea → pagina
    line_to_page = {}
    page_line_offset = 0
    for page in pages:
        page_lines = page.text.split("\n")
        for li in range(len(page_lines)):
            line_to_page[page_line_offset + li] = page.page_num
        page_line_offset += len(page_lines) + 1  # +1 per il separatore

    def _flush_chunk():
        nonlocal current_chunk_lines, current_chunk_start_page
        if current_chunk_lines:
            text = "\n".join(current_chunk_lines).strip()
            if len(text) > 20:
                path_parts = [p for p in [current_section, current_article] if p]
                chunks.append(LegalChunk(
                    chunk_type=current_chunk_type,
                    title=current_title,
                    content=text,
                    markdown=text,  # Simplified: MD = text for now
                    page_start=current_chunk_start_page,
                    page_end=line_to_page.get(len(lines) - 1, current_chunk_start_page),
                    section_path=" > ".join(path_parts) if path_parts else "Corpo",
                    tables=[],
                ))
        current_chunk_lines.clear()

    for line_idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            current_chunk_lines.append(line)
            continue

        page_num = line_to_page.get(line_idx, current_chunk_start_page)

        # Detect sezione
        m = re.match(r'^(?:SEZIONE|PARTE|TITOLO|CAPO)\s+[IVX\d]+[.:\s\-–—]+(.{5,120})', stripped, re.I)
        if m:
            _flush_chunk()
            current_section = stripped[:80]
            current_chunk_type = "sezione"
            current_title = stripped
            current_chunk_start_page = page_num
            current_chunk_lines.append(line)
            continue

        # Detect articolo
        m = re.match(r'^Art(?:icolo)?\.?\s*(\d+(?:[\.\-]\w+)?)\s*[.:\s\-–—]+(.{3,200})', stripped, re.I)
        if m:
            _flush_chunk()
            current_article = f"Art. {m.group(1)} — {m.group(2).strip()[:80]}"
            current_chunk_type = "articolo"
            current_title = current_article
            current_chunk_start_page = page_num
            current_chunk_lines.append(line)
            continue

        # Detect key section headers (no Art prefix but important)
        for kp in _KEY_SECTIONS:
            if re.search(kp, stripped, re.I) and len(stripped) < 200:
                is_header = stripped[0].isupper() or re.match(r'^\d+', stripped)
                if is_header and len(current_chunk_lines) > 3:
                    _flush_chunk()
                    current_chunk_type = "sezione"
                    current_title = stripped[:100]
                    current_chunk_start_page = page_num
                break

        current_chunk_lines.append(line)

    _flush_chunk()

    # Aggiungi tabelle ai chunk più vicini
    for page in pages:
        for tbl in page.tables:
            # Trova il chunk che contiene questa pagina
            for chunk in chunks:
                if chunk.page_start <= page.page_num <= chunk.page_end:
                    chunk.tables.append(tbl)
                    break

    return chunks


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PARSE FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def parse_pdf(pdf_bytes: bytes, filename: str = "document.pdf") -> ParsedDocument:
    """Entry point principale. Parsa il PDF con la migliore strategia disponibile.

    1. Detect nativo vs scansionato
    2. Usa PyMuPDF per estrazione strutturata
    3. Fallback a pdfplumber
    4. Applica chunking semantico
    """
    # Detect tipo
    pdf_info = detect_pdf_type(pdf_bytes)

    # Warn se scansionato
    warnings = []
    if not pdf_info["is_native"]:
        warnings.append("⚠️ PDF scansionato rilevato — l'estrazione potrebbe essere incompleta. Consigliato OCR.")

    # Try PyMuPDF first (migliore per testo strutturato + tabelle + metadati)
    result = _extract_with_fitz(pdf_bytes, filename)
    if result and len(result.full_text.strip()) > 50:
        result.warnings.extend(warnings)
        return result

    # Fallback: pdfplumber
    result = _extract_with_pdfplumber(pdf_bytes, filename)
    if result and len(result.full_text.strip()) > 50:
        result.warnings.extend(warnings)
        return result

    # Ultimo fallback: raw extraction
    return ParsedDocument(
        filename=filename,
        is_native=False,
        total_pages=pdf_info.get("pages", 0),
        full_text="[Impossibile estrarre testo dal PDF]",
        full_markdown="",
        pages=[],
        chunks=[],
        tables_json=[],
        metadata={},
        parser_used="none",
        warnings=["Nessun parser ha potuto estrarre il testo. PDF potrebbe essere corrotto o puramente immagine."],
    )


def get_text_with_tables(parsed: ParsedDocument) -> str:
    """Ritorna il testo completo + tabelle come Markdown inline.
    Questo è il formato ideale da dare al motore AI."""
    if parsed.full_markdown and len(parsed.full_markdown) > len(parsed.full_text):
        return parsed.full_markdown
    return parsed.full_text


def get_page_for_text(parsed: ParsedDocument, search_text: str) -> Optional[int]:
    """Trova in quale pagina si trova un testo specifico."""
    if not search_text or len(search_text) < 3:
        return None
    search_lower = search_text.lower()
    for page in parsed.pages:
        if search_lower in page.text.lower():
            return page.page_num
    # Ricerca parziale (parole significative)
    words = [w for w in search_text.split() if len(w) > 4]
    for page in parsed.pages:
        page_lower = page.text.lower()
        matches = sum(1 for w in words if w.lower() in page_lower)
        if matches >= max(1, len(words) // 2):
            return page.page_num
    return None
