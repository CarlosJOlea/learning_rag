"""Helper utilities for persisting the indexing state."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping

logger = logging.getLogger(__name__)

INDEX_RECORD_FILE = Path("indexed_files.json")

FileSignature = dict[str, float | int]
IndexRecord = dict[str, FileSignature]


def get_file_signature(path: Path | str) -> FileSignature:
    """Return a minimal signature for detecting file modifications."""

    file_path = Path(path)
    stats = file_path.stat()
    return {"size": stats.st_size, "mtime": stats.st_mtime}


def load_index_record() -> IndexRecord:
    """Load the JSON file that stores known file signatures."""

    if not INDEX_RECORD_FILE.exists():
        return {}

    try:
        with INDEX_RECORD_FILE.open("r", encoding="utf-8") as fh:
            raw_data: IndexRecord = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("⚠️ Índice dañado o vacío, se reinicia. (%s)", exc)
        return {}

    return raw_data


def save_index_record(record: Mapping[str, FileSignature]) -> None:
    """Persist the current file signatures to disk."""

    with INDEX_RECORD_FILE.open("w", encoding="utf-8") as fh:
        json.dump(dict(record), fh, indent=4, sort_keys=True)
