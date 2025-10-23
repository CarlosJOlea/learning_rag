import json
import os
import re

# === CONFIGURACIÓN DE RUTA ===
# Sube un nivel desde ingest/ → PythonProject/
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Busca el JSON en el mismo lugar donde lo tienes: pipeline/drug_patterns.json
PATTERNS_FILE = os.path.join(BASE_DIR, "pipeline", "drug_patterns.json")


# === Cargar patrones ===
def _load_patterns():
    """Carga el archivo de patrones o devuelve un esquema vacío si no existe."""
    if not os.path.exists(PATTERNS_FILE):
        print(f"⚠️ No se encontró {PATTERNS_FILE}, se usarán patrones vacíos.")
        return {"drugs": {}, "labs": {}}

    try:
        with open(PATTERNS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"⚠️ Error leyendo {PATTERNS_FILE}, se ignorarán patrones.")
        return {"drugs": {}, "labs": {}}


PATTERNS = _load_patterns()


# === Funciones auxiliares ===
def detect_entity(text: str, patterns: dict, filename: str = "") -> str:
    """
    Busca coincidencias en el texto o nombre de archivo contra una lista de patrones.
    Retorna el nombre normalizado (clave del JSON) o 'desconocido' si no hay match.
    """
    text_low = text.lower()
    fname_low = filename.lower()

    for entity, variants in patterns.items():
        for variant in variants:
            variant_low = variant.lower()

            # Coincidencia directa
            if variant_low in text_low or variant_low in fname_low:
                return entity

            # Coincidencia de palabra completa
            if re.search(rf"\b{re.escape(variant_low)}\b", text_low):
                return entity

    return "desconocido"


# === Detectar medicamento ===
def detect_drug_name(text: str, filename: str = "") -> str:
    """Detecta el medicamento usando patrones configurables."""
    drug = detect_entity(text, PATTERNS.get("drugs", {}), filename)

    # Fallback: si el patrón falla, busca encabezados tipo '# ACTRON®'
    if drug == "desconocido":
        match = re.search(r"#+\s*([A-ZÁÉÍÓÚÑ0-9\-® ]{3,})", text)
        if match:
            return match.group(1).strip("® ").title()

    return drug


# === Detectar laboratorio ===
def detect_lab_name(text: str) -> str:
    """Detecta el laboratorio usando patrones configurables."""
    lab = detect_entity(text, PATTERNS.get("labs", {}))

    # Fallback: frases como “Laboratorios Bayer” o “Fabricado por…”
    if lab == "desconocido":
        match = re.search(
            r"(laboratorios?|fabricado por|distribuido por)[:\s]*([A-ZÁÉÍÓÚÑa-z0-9 .,&\-]+)",
            text, re.IGNORECASE
        )
        if match:
            lab = match.group(2).strip().upper()

    return lab or "desconocido"
