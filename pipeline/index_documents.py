"""Utilities for indexing documents into the local Chroma database."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embeddings.embeddings_factory import load_embeddings
from ingest.loaders import load_document
from ingest.metadata import detect_drug_name, detect_lab_name
from ingest.sections import group_sections, split_by_sections
from ingest.utils import get_file_signature, load_index_record, save_index_record

logger = logging.getLogger(__name__)

DOCS_DIR = Path("docs")
CHROMA_PATH = Path("chroma_db")


@dataclass(frozen=True)
class FileUpdate:
    """Represent a document that needs to be indexed."""

    name: str
    path: Path
    signature: dict[str, float | int]


def discover_updates(index: dict[str, dict[str, float | int]]) -> list[FileUpdate]:
    """Return files whose signature changed compared to the stored index."""

    if not DOCS_DIR.exists():
        logger.warning("⚠️ El directorio de documentos %s no existe.", DOCS_DIR)
        return []

    updates: list[FileUpdate] = []
    for path in sorted(DOCS_DIR.iterdir()):
        if not path.is_file():
            continue

        signature = get_file_signature(path)
        if index.get(path.name) != signature:
            updates.append(FileUpdate(path.name, path, signature))

    return updates


def build_chunks(
    docs: Sequence[Document],
    filename: str,
    splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    """Convert raw documents into normalized LangChain chunks."""

    chunks: list[Document] = []
    timestamp = datetime.now(timezone.utc).isoformat()

    for doc in docs:
        sections = group_sections(split_by_sections(doc.page_content))
        if not sections:
            continue

        drug = detect_drug_name(doc.page_content, filename)
        lab = detect_lab_name(doc.page_content)

        for section in sections:
            content = section["content"].strip()
            if not content:
                continue

            for fragment in splitter.split_text(content):
                metadata = {
                    "section": section["section"],
                    "drug": drug,
                    "lab": lab,
                    "filename": filename,
                    "indexed_at": timestamp,
                }

                header = (
                    f"Medicamento: {drug}\n"
                    f"Laboratorio: {lab}\n"
                    f"Sección: {section['section']}\n\n"
                )

                chunks.append(Document(page_content=header + fragment, metadata=metadata))

    return chunks


def run_index_pipeline() -> None:
    """Index new or modified documents into the persistent Chroma database."""

    logger.info("📂 Escaneando carpeta de documentos...")

    index = load_index_record()
    updates = discover_updates(index)

    if not updates:
        logger.info("✅ No hay cambios, la base vectorial ya está actualizada.")
        return

    embedding_function = load_embeddings()
    db = Chroma(persist_directory=str(CHROMA_PATH), embedding_function=embedding_function)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],
    )

    total_chunks = 0

    for update in updates:
        logger.info("\n📄 Procesando: %s", update.name)

        try:
            docs = load_document(str(update.path))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("❌ Error procesando %s: %s", update.name, exc)
            continue

        if not docs:
            logger.warning("⚠️ %s: sin texto válido.", update.name)
            continue

        chunks = build_chunks(docs, update.name, splitter)
        if not chunks:
            logger.warning("⚠️ %s: sin fragmentos tras el procesamiento.", update.name)
            continue

        try:
            db.add_documents(chunks)
        except Exception as exc:  # pragma: no cover - depende de backend externo
            logger.exception("❌ Error al almacenar %s: %s", update.name, exc)
            continue

        index[update.name] = update.signature
        total_chunks += len(chunks)
        logger.info("✅ %s: %s fragmentos indexados.", update.name, len(chunks))

    save_index_record(index)
    logger.info("\n📚 Total de fragmentos añadidos: %s", total_chunks)
    logger.info("✅ Base vectorial actualizada correctamente.")
