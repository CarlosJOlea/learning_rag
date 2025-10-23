import os, json

INDEX_RECORD_FILE = "indexed_files.json"

def get_file_signature(path):
    stats = os.stat(path)
    return {"size": stats.st_size, "mtime": stats.st_mtime}

def load_index_record():
    if not os.path.exists(INDEX_RECORD_FILE):
        return {}
    try:
        with open(INDEX_RECORD_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        print("⚠️ Índice dañado o vacío, se reinicia.")
        return {}

def save_index_record(record):
    with open(INDEX_RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=4)
