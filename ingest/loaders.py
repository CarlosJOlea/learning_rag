from langchain_community.document_loaders import (
    PyPDFLoader, PyMuPDFLoader, TextLoader,
    UnstructuredFileLoader, UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader, CSVLoader
)
from .ocr import load_pdf_with_ocr

def load_document(file_path):
    ext = file_path.lower().split(".")[-1]
    loaders = {
        "pdf": lambda p: PyMuPDFLoader(p).load() or load_pdf_with_ocr(p),
        "txt": lambda p: TextLoader(p, encoding="utf-8").load(),
        "md": lambda p: UnstructuredMarkdownLoader(p).load(),
        "html": lambda p: UnstructuredHTMLLoader(p).load(),
        "htm": lambda p: UnstructuredHTMLLoader(p).load(),
        "csv": lambda p: CSVLoader(p).load(),
        "docx": lambda p: UnstructuredFileLoader(p).load(),
    }

    if ext not in loaders:
        print(f"⚠️ Tipo no soportado: {file_path}")
        return []
    try:
        return loaders[ext](file_path)
    except Exception as e:
        print(f"❌ Error al cargar {file_path}: {e}")
        return []
