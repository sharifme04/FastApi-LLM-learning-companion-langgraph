"""PDF parsing using pymupdf (fitz).

Extracts text per page and returns a list of (page_number, text) tuples.
Caller decides how to chunk across or within pages.
"""

import logging
from io import BytesIO

from app.utils.exceptions import IngestionError

logger = logging.getLogger("learning_companion")


def extract_pdf_pages(content: bytes) -> list[tuple[int, str]]:
    """Extract text from each page of a PDF.

    Args:
        content: Raw PDF bytes.

    Returns:
        List of (1-based page number, page text) tuples. Empty pages are kept
        so chunk-page mapping stays accurate.
    """
    try:
        import fitz  # pymupdf
    except ImportError as e:
        raise IngestionError(f"pymupdf not installed: {e}")

    try:
        doc = fitz.open(stream=BytesIO(content), filetype="pdf")
    except Exception as e:
        raise IngestionError(f"Failed to open PDF: {e}")

    pages: list[tuple[int, str]] = []
    try:
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            pages.append((i + 1, text.strip()))
    finally:
        doc.close()

    total_chars = sum(len(t) for _, t in pages)
    if total_chars < 100:
        raise IngestionError(
            "Extracted text is too short — PDF may be a scanned image (OCR required).",
            detail={"chars_extracted": total_chars, "page_count": len(pages)},
        )

    logger.info(
        "PDF parsed",
        extra={"pages": len(pages), "chars": total_chars},
    )
    return pages
