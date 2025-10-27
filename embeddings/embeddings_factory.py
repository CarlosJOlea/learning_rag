"""Factory helpers for creating embedding models."""

from __future__ import annotations

import logging
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)


def load_embeddings() -> Any:
    """Instantiate the preferred embedding model with a safe fallback."""

    try:
        logger.info("🧠 Embeddings: Ollama (nomic-embed-text)")
        return OllamaEmbeddings(model="nomic-embed-text")
    except Exception as exc:  # pragma: no cover - depende de servicios externos
        logger.warning("⚠️ Fallback a MiniLM (%s)", exc)
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
