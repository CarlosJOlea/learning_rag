from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from langchain_core.documents import Document
import os

def load_pdf_with_ocr(pdf_path):
    print(f"🟡 OCR: {os.path.basename(pdf_path)}")
    try:
        pages = convert_from_path(pdf_path)
    except Exception as e:
        print(f"❌ Error al convertir {pdf_path}: {e}")
        return []

    docs = []
    for i, img in enumerate(pages):
        text = pytesseract.image_to_string(img, lang="spa+eng").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"page": i + 1}))
    return docs
