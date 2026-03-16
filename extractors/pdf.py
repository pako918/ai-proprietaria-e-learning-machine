"""Estrazione testo da PDF."""

import re
import pdfplumber


def _decode_cid(text: str) -> str:
    """Decodifica caratteri (cid:XX) presenti in alcuni PDF mal codificati."""
    def _repl(m):
        try:
            return chr(int(m.group(1)))
        except (ValueError, OverflowError):
            return ""
    decoded = re.sub(r"\(cid:(\d+)\)", _repl, text)
    decoded = re.sub(r"  +", " ", decoded)
    return decoded


def extract_text_from_pdf(pdf_path: str, max_pages: int | None = None) -> str:
    """Estrae tutto il testo da un PDF disciplinare."""
    text_parts = []
    has_cid = False
    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages[:max_pages] if max_pages else pdf.pages
        for i, page in enumerate(pages):
            page_text = page.extract_text() or ""
            if "(cid:" in page_text:
                has_cid = True
                page_text = _decode_cid(page_text)
            if page_text.strip():
                text_parts.append(f"--- Pagina {i+1} ---\n{page_text}")

            tables_found = []
            for strat in [{"vertical_strategy": "lines", "horizontal_strategy": "lines"}, {}]:
                try:
                    tables = page.extract_tables(strat) if strat else page.extract_tables()
                except Exception:
                    continue
                if tables:
                    tables_found = tables
                    break
            for table in tables_found:
                if not table:
                    continue
                rows = []
                for row in table:
                    clean = [str(cell or "").strip().replace("\n", " ") for cell in row]
                    if any(c for c in clean):
                        rows.append(clean)
                if rows:
                    table_str = "\n".join(" | ".join(cell for cell in row) for row in rows)
                    if "(cid:" in table_str:
                        table_str = _decode_cid(table_str)
                    text_parts.append(f"[TABELLA pag.{i+1}]\n{table_str}")

    result = "\n\n".join(text_parts)
    if has_cid:
        result = re.sub(r"\n{3,}", "\n\n", result)
    return result
