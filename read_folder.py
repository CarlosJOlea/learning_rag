import os
import re
import json
from datetime import datetime
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pdf2image import convert_from_path
from PIL import Image
from langchain_ollama import OllamaEmbeddings

import pytesseract

# === CONFIGURACIÓN ===
DOCS_DIR = "docs"
CHROMA_PATH = "chroma_db"
INDEX_RECORD_FILE = "indexed_files.json"

# === SECCIONES MÉDICAS MEJORADAS ===
MEDICAL_SECTIONS = [
    "COMPOSICIÓN", "FORMA FARMACÉUTICA", "INDICACIONES", "CONTRAINDICACIONES",
    "PRECAUCIONES", "INTERACCIONES", "DOSIS", "REACCIONES ADVERSAS",
    "PRESENTACIONES", "ALMACENAMIENTO", "FARMACOCINÉTICA"
]

# === OCR AUXILIAR ===
def load_pdf_with_ocr(pdf_path):
    print(f"🟡 Aplicando OCR a: {os.path.basename(pdf_path)} ...")
    try:
        pages = convert_from_path(pdf_path)
    except Exception as e:
        print(f"❌ No se pudo convertir {pdf_path} a imágenes: {e}")
        return []

    ocr_docs = []
    for i, img in enumerate(pages):
        text = pytesseract.image_to_string(img, lang="spa+eng")
        if text.strip():
            ocr_docs.append(Document(page_content=text, metadata={"page": i + 1}))
    return ocr_docs


# === FUNCIONES DE INDEXACIÓN ===
def get_file_signature(file_path):
    stats = os.stat(file_path)
    return {"size": stats.st_size, "mtime": stats.st_mtime}


def load_index_record():
    if os.path.exists(INDEX_RECORD_FILE):
        try:
            with open(INDEX_RECORD_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                return json.loads(content) if content else {}
        except json.JSONDecodeError:
            print("⚠️ Archivo de índice dañado o vacío, se reiniciará.")
            return {}
    return {}


def save_index_record(record):
    with open(INDEX_RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=4)


# === DETECCIÓN DE METADATOS ===
def detect_drug_name(text: str, filename: str) -> str:
    text_low = text.lower()
    fname = filename.lower()

    drug_patterns = {
        "libertrim": ["libertrim", "trimebutina"],
        "ampiquim": ["ampi-quim", "ampiquim", "ampicilina"],
        "abacavir": ["abacavir"]
    }

    for drug, patterns in drug_patterns.items():
        if any(p in text_low or p in fname for p in patterns):
            return drug
    return "desconocido"


def detect_lab_name(text: str) -> str:
    text_low = text.lower()
    if "a.f. laboratorios" in text_low or "aplicaciones farmacéuticas" in text_low:
        return "A.F. LABORATORIOS APLICACIONES FARMACÉUTICAS"
    if "química y farmacia" in text_low:
        return "QUÍMICA Y FARMACIA"
    if "diba" in text_low:
        return "LABORATORIOS DIBA"
    return "desconocido"


# === SEPARADOR POR SECCIONES ===
SECTION_KEYWORDS = [
    "FORMA FARMACÉUTICA", "FORMULACIÓN", "INDICACIONES", "FARMACOCINÉTICA",
    "CONTRAINDICACIONES", "PRECAUCIONES", "REACCIONES", "ADVERSAS",
    "DOSIS", "PRESENTACIONES", "RECOMENDACIONES", "ALMACENAMIENTO"
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


# === AGRUPADOR DE SECCIONES ===
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


# === TEXT SPLITTER ===
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " "]
)


# === EMBEDDINGS ===
def load_embedding_function():
    try:
        print("🧠 Usando embeddings locales de Ollama: nomic-embed-text ...")
        return OllamaEmbeddings(model="nomic-embed-text")
    except Exception as e:
        print(f"⚠️ No se pudo cargar Ollama, usando fallback MiniLM. ({e})")
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# === PROCESAMIENTO PRINCIPAL ===
print("📂 Leyendo carpeta de documentos...")

indexed_files = load_index_record()
new_files = []

for filename in os.listdir(DOCS_DIR):
    file_path = os.path.join(DOCS_DIR, filename)
    if not os.path.isfile(file_path):
        continue

    sig = get_file_signature(file_path)
    old_sig = indexed_files.get(filename)

    if old_sig == sig:
        print(f"⏩ {filename}: sin cambios, se omite.")
        continue

    new_files.append((filename, file_path, sig))

print(f"🆕 {len(new_files)} archivos nuevos o modificados para indexar.")
if not new_files:
    print("✅ No hay archivos nuevos, la base ya está actualizada.")
    exit()

# === Base vectorial ===
embedding_function = load_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

all_new_docs = []

# === PROCESAR ARCHIVOS ===
for filename, file_path, sig in new_files:
    try:
        # Seleccionar loader adecuado
        if filename.endswith(".pdf"):
            try:
                loader = PyMuPDFLoader(file_path)
                docs = loader.load()
                if len(docs) <= 1:
                    docs = load_pdf_with_ocr(file_path)
            except Exception:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
        elif filename.endswith(".docx"):
            loader = UnstructuredFileLoader(file_path)
            docs = loader.load()
        elif filename.endswith(".md"):
            loader = UnstructuredMarkdownLoader(file_path)
            docs = loader.load()
        elif filename.endswith((".html", ".htm")):
            loader = UnstructuredHTMLLoader(file_path)
            docs = loader.load()
        elif filename.endswith(".csv"):
            loader = CSVLoader(file_path)
            docs = loader.load()
        else:
            print(f"⚠️ Tipo de archivo no soportado: {filename}")
            continue

        # === SPLIT, AGRUPACIÓN Y METADATOS ===
        all_chunks = []
        for d in docs:
            raw_sections = split_by_sections(d.page_content)
            grouped_sections = group_sections(raw_sections, window_size=1)

            for chunk in grouped_sections:
                if not chunk["content"].strip():
                    continue

                drug = detect_drug_name(d.page_content, filename)
                lab = detect_lab_name(d.page_content)

                # Sub-chunks por tokens
                subchunks = splitter.split_text(chunk["content"])
                for subtext in subchunks:
                    metadata = {
                        "section": chunk["section"],
                        "drug": drug,
                        "lab": lab,
                        "filename": filename,
                        "indexed_at": datetime.now().isoformat()
                    }

                    # Añadir encabezado contextual
                    context_header = (
                        f"Medicamento: {drug}\n"
                        f"Laboratorio: {lab}\n"
                        f"Sección: {chunk['section']}\n\n"
                    )

                    all_chunks.append(Document(
                        page_content=context_header + subtext,
                        metadata=metadata
                    ))

        # === Añadir a la base ===
        if all_chunks:
            db.add_documents(all_chunks)
            all_new_docs.extend(all_chunks)
            indexed_files[filename] = sig
            print(f"✅ {filename}: {len(all_chunks)} fragmentos enriquecidos indexados.")
        else:
            print(f"⚠️ {filename}: no se detectaron secciones válidas.")

    except Exception as e:
        print(f"❌ Error procesando {filename}: {e}")

# === Guardar registro ===
save_index_record(indexed_files)
print(f"📚 Se agregaron {len(all_new_docs)} nuevos fragmentos enriquecidos.")
print("✅ Base vectorial actualizada correctamente.")