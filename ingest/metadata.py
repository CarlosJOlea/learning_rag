"""Entity detection helpers based on configurable pattern files."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PATTERNS_FILE = BASE_DIR / "pipeline" / "drug_patterns.json"


def _load_patterns() -> Dict[str, Dict[str, list[str]]]:
    """Load entity patterns from disk, returning empty defaults on error."""

    if not PATTERNS_FILE.exists():
        logger.warning(
            "⚠️ No se encontró %s, se usarán patrones vacíos.", PATTERNS_FILE
        )
        return {"drugs": {}, "labs": {}}

    try:
        with PATTERNS_FILE.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        logger.warning(
            "⚠️ Error leyendo %s, se ignorarán patrones. (%s)", PATTERNS_FILE, exc
        )
        return {"drugs": {}, "labs": {}}


PATTERNS = _load_patterns()


def detect_entity(text: str, patterns: Dict[str, list[str]], filename: str = "") -> str:
    """Return the normalized entity name matching the provided text."""

    text_low = text.lower()
    fname_low = filename.lower()

    for entity, variants in patterns.items():
        for variant in variants:
            variant_low = variant.lower()

            if variant_low in text_low or variant_low in fname_low:
                return entity

            if re.search(rf"\b{re.escape(variant_low)}\b", text_low):
                return entity

    return "desconocido"


def detect_drug_name(text: str, filename: str = "") -> str:
    """Detect the drug name, falling back to Markdown headings when needed."""

    drug = detect_entity(text, PATTERNS.get("drugs", {}), filename)

    if drug == "desconocido":
        match = re.search(r"#+\s*([A-ZÁÉÍÓÚÑ0-9\-® ]{3,})", text)
        if match:
            return match.group(1).strip("® ").title()

    return drug


def detect_lab_name(text: str) -> str:
    """Detect the lab name, using heuristics when patterns fail."""

    lab = detect_entity(text, PATTERNS.get("labs", {}))

    if lab == "desconocido":
        match = re.search(
            r"(laboratorios?|fabricado por|distribuido por)[:\s]*([A-ZÁÉÍÓÚÑa-z0-9 .,&\-]+)",
            text,
            re.IGNORECASE,
        )
        if match:
            lab = match.group(2).strip().upper()

    return lab or "desconocido"
