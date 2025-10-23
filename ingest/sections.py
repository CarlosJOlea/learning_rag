import re

SECTION_KEYWORDS = [
    "FORMA FARMACÉUTICA", "FORMULACIÓN", "COMPOSICIÓN", "INDICACIONES",
    "FARMACOCINÉTICA", "FARMACODINAMIA", "CONTRAINDICACIONES", "PRECAUCIONES",
    "REACCIONES", "ADVERSAS", "INTERACCIONES", "DOSIS", "ADMINISTRACIÓN",
    "SOBREDOSIFICACIÓN", "PRESENTACIONES", "ALMACENAMIENTO", "FABRICANTE",
    "LEYENDAS", "USO EN EMBARAZO", "LACTANCIA"
]

def split_by_sections(text):
    sections = []
    current_section = "general"
    buffer = []

    for line in text.splitlines():
        for kw in SECTION_KEYWORDS:
            if re.search(rf"\b{kw}\b", line, re.IGNORECASE):
                if buffer:
                    sections.append({
                        "section": current_section.lower(),
                        "content": "\n".join(buffer).strip()
                    })
                    buffer = []
                current_section = kw
                break
        else:
            buffer.append(line)

    if buffer:
        sections.append({
            "section": current_section.lower(),
            "content": "\n".join(buffer).strip()
        })
    return sections

def group_sections(sections, window_size=2):
    grouped = []
    for i in range(0, len(sections), window_size):
        combined = "\n\n".join(s["content"] for s in sections[i:i + window_size])
        merged_sections = [s["section"] for s in sections[i:i + window_size]]
        grouped.append({
            "section": "+".join(merged_sections),
            "content": combined
        })
    return grouped
