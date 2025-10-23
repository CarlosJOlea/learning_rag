from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma_db", embedding_function=embedding_function)

# === Obtener todos los metadatos ===
all_data = db.get(include=["metadatas", "documents"])

print(f"🔍 Total de fragmentos en la base: {len(all_data['documents'])}")

for i, meta in enumerate(all_data["metadatas"][:100]):
    print(f"\n🧩 Fragmento {i+1}")
    print("Documento:", all_data["documents"][i][:120].replace("\n", " "))
    print("Metadatos:", meta)
