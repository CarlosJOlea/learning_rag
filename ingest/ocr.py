"""OCR helpers used as a fallback for PDF ingestion."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pytesseract
from langchain_core.documents import Document
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)


def load_pdf_with_ocr(pdf_path: str | Path) -> List[Document]:
    """Extract text from a PDF using Tesseract OCR."""

    path = Path(pdf_path)
    logger.info("🟡 OCR: %s", path.name)

    try:
        pages = convert_from_path(str(path))
    except Exception as exc:  # pragma: no cover - depende de backend externo
        logger.error("❌ Error al convertir %s: %s", path.name, exc)
        return []

    docs: List[Document] = []
    for index, image in enumerate(pages, start=1):
        text = pytesseract.image_to_string(image, lang="spa+eng").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"page": index}))

    return docs
