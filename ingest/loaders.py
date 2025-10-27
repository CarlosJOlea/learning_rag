"""Document loader helpers used by the ingestion pipeline."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Dict, List

from langchain_community.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

from .ocr import load_pdf_with_ocr

logger = logging.getLogger(__name__)

LoaderFn = Callable[[Path], List[Document]]


def _load_pdf(path: Path) -> List[Document]:
    """Load a PDF using PyMuPDF, falling back to OCR when needed."""

    try:
        pages = PyMuPDFLoader(str(path)).load()
    except Exception as exc:  # pragma: no cover - depende de backend externo
        logger.warning("⚠️ Error al leer %s con PyMuPDF: %s", path.name, exc)
        pages = []

    if pages:
        return pages

    ocr_pages = load_pdf_with_ocr(str(path))
    if ocr_pages:
        return ocr_pages

    logger.warning("⚠️ %s: no se pudo extraer texto del PDF.", path.name)
    return []


LOADERS: Dict[str, LoaderFn] = {
    "pdf": _load_pdf,
    "txt": lambda path: TextLoader(str(path), encoding="utf-8").load(),
    "md": lambda path: UnstructuredMarkdownLoader(str(path)).load(),
    "html": lambda path: UnstructuredHTMLLoader(str(path)).load(),
    "htm": lambda path: UnstructuredHTMLLoader(str(path)).load(),
    "csv": lambda path: CSVLoader(str(path)).load(),
    "docx": lambda path: UnstructuredFileLoader(str(path)).load(),
}


def load_document(file_path: str | Path) -> List[Document]:
    """Load a document using the appropriate LangChain loader."""

    path = Path(file_path)
    ext = path.suffix.lower().lstrip(".")

    loader = LOADERS.get(ext)
    if loader is None:
        logger.warning("⚠️ Tipo no soportado: %s", path.name)
        return []

    try:
        return loader(path)
    except Exception as exc:  # pragma: no cover - depende de backend externo
        logger.error("❌ Error al cargar %s: %s", path.name, exc)
        return []
