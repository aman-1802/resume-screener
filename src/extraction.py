"""Hybrid PDF text extraction: direct text layer first, local OCR fallback.

Most resumes and JDs are digital PDFs with an embedded text layer, so
pdfplumber extraction is instant and 100% accurate. Only scanned/image-based
PDFs (sparse or empty text layer) fall back to rendering pages to images and
running them through the local Umi-OCR HTTP service.
"""

from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

from . import ocr_client

# Below this many characters per page, treat the text layer as unreliable
# and fall back to OCR (catches scanned pages with a few stray characters).
MIN_CHARS_PER_PAGE = 40


def _direct_extract(pdf_path: Path) -> str:
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
    return "\n".join(texts)


def _ocr_extract(pdf_path: Path, dpi: int = 200) -> str:
    doc = fitz.open(pdf_path)
    texts = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            image_bytes = pix.tobytes("png")
            texts.append(ocr_client.ocr_image_bytes(image_bytes))
    finally:
        doc.close()
    return "\n".join(texts)


def extract_text(pdf_path: str | Path) -> str:
    """Return the best-effort plain text for a PDF, using OCR only if needed."""
    pdf_path = Path(pdf_path)

    direct_text = _direct_extract(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)

    avg_chars_per_page = len(direct_text) / max(page_count, 1)

    if avg_chars_per_page >= MIN_CHARS_PER_PAGE:
        return direct_text

    # Sparse/empty text layer -> likely scanned, fall back to local OCR.
    return _ocr_extract(pdf_path)
