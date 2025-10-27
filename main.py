"""Command line entry-point for the indexing pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from pipeline.index_documents import run_index_pipeline


def main() -> None:
    """Configure logging and execute the indexing pipeline."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("pdfminer").setLevel(logging.WARNING)
    logging.info("📁 Working directory: %s", Path.cwd())
    run_index_pipeline()


if __name__ == "__main__":
    main()
