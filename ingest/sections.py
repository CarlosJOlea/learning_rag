"""Utilities to extract and group logical sections from documents."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

Section = dict[str, str]

SECTION_KEYWORDS: Sequence[str] = (
    "FORMA FARMACÉUTICA",
    "FORMULACIÓN",
    "COMPOSICIÓN",
    "INDICACIONES",
    "FARMACOCINÉTICA",
    "FARMACODINAMIA",
    "CONTRAINDICACIONES",
    "PRECAUCIONES",
    "REACCIONES",
    "ADVERSAS",
    "INTERACCIONES",
    "DOSIS",
    "ADMINISTRACIÓN",
    "SOBREDOSIFICACIÓN",
    "PRESENTACIONES",
    "ALMACENAMIENTO",
    "FABRICANTE",
    "LEYENDAS",
    "USO EN EMBARAZO",
    "LACTANCIA",
)


def split_by_sections(text: str) -> List[Section]:
    """Split text into sections based on known keywords."""

    sections: List[Section] = []
    current_section = "general"
    buffer: List[str] = []

    for line in text.splitlines():
        for keyword in SECTION_KEYWORDS:
            if re.search(rf"\b{keyword}\b", line, re.IGNORECASE):
                if buffer:
                    sections.append(
                        {
                            "section": current_section.lower(),
                            "content": "\n".join(buffer).strip(),
                        }
                    )
                    buffer = []
                current_section = keyword
                break
        else:
            buffer.append(line)

    if buffer:
        sections.append(
            {
                "section": current_section.lower(),
                "content": "\n".join(buffer).strip(),
            }
        )

    return sections


def group_sections(sections: Iterable[Section], window_size: int = 2) -> List[Section]:
    """Group contiguous sections to provide more context to the splitter."""

    sections_list = list(sections)
    grouped: List[Section] = []

    for i in range(0, len(sections_list), window_size):
        chunk = sections_list[i : i + window_size]
        combined = "\n\n".join(section["content"] for section in chunk)
        merged_names = [section["section"] for section in chunk]
        grouped.append({"section": "+".join(merged_names), "content": combined})

    return grouped
