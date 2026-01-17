import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from answer_generator import generate_answer
from llm_client import call_llm


VECTOR_DB_PATH = "vector_store"

SCHEME_NAME_MAP = {
    "pm_surya_ghar": "PM Surya Ghar: Muft Bijli Yojana",
    "national_solar_mission": "Jawaharlal Nehru National Solar Mission",
    "pm_kusum": "PM-KUSUM Scheme",
    "national_bio_energy": "National Bio-Energy Mission",
    "smart_grid_mission": "National Smart Grid Mission",
    "jal_jeevan_mission": "Jal Jeevan Mission",
    "atal_bhujal_yojana": "Atal Bhujal Yojana",
    "catch_the_rain": "Jal Shakti Abhiyan – Catch the Rain",
    "national_water_mission": "National Water Mission",
    "amrut_mission": "AMRUT Mission"
}


# -------------------- VECTOR DB --------------------

def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def retrieve_context(query, k=3):
    db = load_vector_db()
    return db.similarity_search(query, k=k)


# -------------------- SCHEME NAME --------------------

def get_scheme_name_from_source(doc):
    source = doc.metadata.get("source", "")
    if not source:
        return "Unknown Scheme"

    filename = os.path.basename(source).replace(".txt", "")
    return SCHEME_NAME_MAP.get(
        filename,
        filename.replace("_", " ").title()
    )


# -------------------- VALIDATION --------------------

def is_valid_scheme_query(query):
    query = query.lower().strip()

    # ❌ Block instruction-style inputs
    if query.startswith(("apply ", "submit ", "visit ")):
        return False

    scheme_keywords = [
        "scheme", "yojana", "mission", "subsidy",
        "solar", "water", "energy", "pump", "conservation"
    ]

    question_words = [
        "what", "which", "how", "tell", "explain",
        "details", "benefits", "eligibility", "apply", "application"
    ]

    has_scheme_keyword = any(k in query for k in scheme_keywords)
    has_question_intent = any(q in query for q in question_words)
    is_descriptive_scheme_query = "scheme" in query

    # ✅ Accept if scheme-related AND
    # (question OR short intent OR descriptive scheme phrase)
    return has_scheme_keyword and (
        has_question_intent
        or len(query.split()) <= 6
        or is_descriptive_scheme_query
    )





# -------------------- CONTEXT REFINEMENT --------------------

def refine_context(retrieved_docs):
    if not retrieved_docs:
        return ""

    primary_doc = retrieved_docs[0]
    primary_content = primary_doc.page_content.strip()
    scheme_name = get_scheme_name_from_source(primary_doc)

    refined_context = f"""
Primary Scheme: {scheme_name}

{primary_content}
""".strip()

    return refined_context


# -------------------- MAIN --------------------

if __name__ == "__main__":
    user_query = input("Ask a question: ").strip()

    if not is_valid_scheme_query(user_query):
        print(
            "\n⚠️ Invalid query.\n"
            "Please ask a clear question about a government scheme.\n"
            "Example: 'solar subsidy for home' or 'what is National Water Mission'\n"
        )
        exit()

    results = retrieve_context(user_query)
    clean_context = refine_context(results)

    answer_prompt = generate_answer(user_query, clean_context)
    final_answer = call_llm(answer_prompt)

    print("\n🤖 AI Answer (Citizen-Friendly):\n")
    print(final_answer)
