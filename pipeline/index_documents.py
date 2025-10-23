import os
from datetime import datetime
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingest.utils import load_index_record, save_index_record, get_file_signature
from ingest.loaders import load_document
from ingest.metadata import detect_drug_name, detect_lab_name
from ingest.sections import split_by_sections, group_sections
from embeddings.embeddings_factory import load_embeddings

DOCS_DIR = "docs"
CHROMA_PATH = "chroma_db"


def run_index_pipeline():
    print("📂 Escaneando carpeta de documentos...")

    # Cargar índice existente
    index = load_index_record()
    new_files = []

    # Detectar archivos nuevos o modificados
    for fname in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, fname)
        if not os.path.isfile(path):
            continue

        sig = get_file_signature(path)
        if index.get(fname) != sig:
            new_files.append((fname, path, sig))

    if not new_files:
        print("✅ No hay cambios, la base vectorial ya está actualizada.")
        return

    # Inicializar embeddings y base vectorial
    embedding_function = load_embeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )

    total_chunks = []

    # Procesar cada archivo nuevo o modificado
    for fname, path, sig in new_files:
        print(f"\n📄 Procesando: {fname}")
        try:
            docs = load_document(path)
            chunks = []

            for d in docs:
                sections = group_sections(split_by_sections(d.page_content))
                for sec in sections:
                    if not sec["content"].strip():
                        continue

                    subtexts = splitter.split_text(sec["content"])
                    for sub in subtexts:
                        drug = detect_drug_name(d.page_content, fname)
                        lab = detect_lab_name(d.page_content)

                        metadata = {
                            "section": sec["section"],
                            "drug": drug,
                            "lab": lab,
                            "filename": fname,
                            "indexed_at": datetime.now().isoformat()
                        }

                        header = (
                            f"Medicamento: {drug}\n"
                            f"Laboratorio: {lab}\n"
                            f"Sección: {sec['section']}\n\n"
                        )

                        chunks.append(Document(
                            page_content=header + sub,
                            metadata=metadata
                        ))

            if chunks:
                db.add_documents(chunks)
                total_chunks.extend(chunks)
                index[fname] = sig
                print(f"✅ {fname}: {len(chunks)} fragmentos indexados.")
            else:
                print(f"⚠️ {fname}: sin texto válido.")

        except Exception as e:
            print(f"❌ Error procesando {fname}: {e}")

    # Guardar registro actualizado
    save_index_record(index)
    print(f"\n📚 Total de fragmentos añadidos: {len(total_chunks)}")
    print("✅ Base vectorial actualizada correctamente.")
