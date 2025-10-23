from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma  # ✅ versión moderna
from langchain_core.prompts import ChatPromptTemplate
import json
import os

# === CONFIGURACIÓN ===
CHROMA_PATH = "chroma_db"
BASE_DIR = os.path.dirname(__file__)
PATTERNS_FILE = os.path.join(BASE_DIR, "pipeline", "drug_patterns.json")

print("🧠 Cargando base vectorial...")

# === Cargar embeddings y base ===
embedding_function = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 40, "lambda_mult": 0.5}
)

# === Modelo local (Gemma 2B vía Ollama) ===
llm = OllamaLLM(model="gemma:2b")

# === Expansión de sinónimos ===
SYN_EXPANSIONS = {
    "presentaciones": ["presentación", "formulación", "envase", "tabletas", "cápsulas", "mg", "ml"],
    "indicaciones": ["uso terapéutico", "tratamiento", "para qué sirve", "aplicación"],
    "contraindicaciones": ["advertencias", "restricciones", "precauciones"],
    "efectos adversos": ["reacciones adversas", "efectos secundarios"],
    "dosis": ["posología", "administración", "modo de uso"],
}

def expand_query(q: str) -> str:
    ql = q.lower()
    terms = []
    for k, syns in SYN_EXPANSIONS.items():
        if k in ql:
            terms += syns
    return q + " " + " ".join(set(terms))


# === Prompt ===
prompt = ChatPromptTemplate.from_template("""
Eres un asistente farmacéutico. Responde SOLO con información explícita en el contexto.

Si no hay información clara o se habla de otro medicamento, responde: **NO_ENCONTRADO**.

❓ Pregunta: {question}

📚 Contexto relevante:
{context}

🔒 Reglas:
- No inventes ni infieras nada fuera del contexto.
- Si el contexto menciona otro fármaco, responde NO_ENCONTRADO.
- Si no hay datos sobre la sección solicitada (p. ej., dosis o contraindicaciones), responde NO_ENCONTRADO.
""")

# === Parámetros ===
MIN_CONTEXT_CHARS = 50  # ⚡ más permisivo
KNOWN_DRUGS = ["libertrim", "ampiquim", "abacavir", "trimebutina", "ampicilina", "ibuprofeno", "actron"]

def extract_drug_from_question(q: str):
    ql = q.lower()
    for drug in KNOWN_DRUGS:
        if drug in ql:
            return drug
    return None


# === Cargar equivalencias desde pipeline/drug_patterns.json ===
def load_drug_equivalents():
    """Carga las equivalencias genérico↔comercial desde el JSON."""
    if not os.path.exists(PATTERNS_FILE):
        print(f"⚠️ No se encontró {PATTERNS_FILE}, se usarán equivalencias vacías.")
        return {}
    with open(PATTERNS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    drugs = data.get("drugs", {})
    equivalents = {}
    for canonical, variants in drugs.items():
        canonical_low = canonical.lower().strip()
        equivalents.setdefault(canonical_low, [])
        for v in variants:
            v_low = v.lower().strip()
            # agrega relación bidireccional
            equivalents[canonical_low].append(v_low)
            equivalents.setdefault(v_low, []).append(canonical_low)
    return equivalents

DRUG_EQUIVALENTS = load_drug_equivalents()


def normalize_drug_name(name: str) -> str:
    return name.lower().strip() if name else ""


def drugs_equivalent(d1: str, d2: str) -> bool:
    """Determina si dos nombres son equivalentes (genérico ↔ comercial)."""
    if not d1 or not d2:
        return False
    d1, d2 = normalize_drug_name(d1), normalize_drug_name(d2)
    if d1 == d2:
        return True
    eq1 = DRUG_EQUIVALENTS.get(d1, [])
    return d2 in eq1


def docs_match_drug(docs, drug: str) -> bool:
    """Verifica si al menos un documento coincide con el fármaco o su equivalente."""
    if not drug:
        return True
    for d in docs:
        md = getattr(d, "metadata", {}) or {}
        doc_drug = normalize_drug_name(md.get("drug") or "")
        if drugs_equivalent(drug, doc_drug):
            return True
    return False


def has_sufficient_context(docs):
    if not docs:
        return False
    combined = "".join([d.page_content for d in docs if getattr(d, "page_content", "").strip()])
    return len(combined) >= MIN_CONTEXT_CHARS


# === Función principal ===
def answer_query(question: str):
    expanded_question = expand_query(question)
    docs = retriever.invoke(expanded_question)

    if not has_sufficient_context(docs):
        return "NO_ENCONTRADO"

    queried_drug = extract_drug_from_question(question)
    if queried_drug and not docs_match_drug(docs, queried_drug):
        return "NO_ENCONTRADO"

    # Combina contexto
    context = "\n\n".join([d.page_content for d in docs])
    if len(context) > 32000:
        context = context[:32000]

    if not context.strip():
        return "NO_ENCONTRADO"

    formatted_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(formatted_prompt)

    # Filtro adicional más flexible
    resp_low = response.lower().strip()
    if resp_low in ["no_encontrado", "no encontrado"] or \
       ("no hay información" in resp_low or "no existen datos" in resp_low):
        return "NO_ENCONTRADO"

    return response.strip()


# === Bucle interactivo ===
print("✅ Sistema listo. Escribe tu pregunta o 'salir' para terminar.\n")

while True:
    q = input("❓ Pregunta: ")
    if q.lower() == "salir":
        break
    print("\n🧩 Respuesta:")
    print(answer_query(q))
    print("-" * 80)
