from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

def load_embeddings():
    try:
        print("🧠 Embeddings: Ollama (nomic-embed-text)")
        return OllamaEmbeddings(model="nomic-embed-text")
    except Exception as e:
        print(f"⚠️ Fallback a MiniLM ({e})")
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
