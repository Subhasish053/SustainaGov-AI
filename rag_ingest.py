import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS



DATA_PATH = "data"
VECTOR_DB_PATH = "vector_store"


def load_documents():
    documents = []
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".txt"):
                loader = TextLoader(
                    os.path.join(root, file),
                    encoding="utf-8"
                )
                documents.extend(loader.load())
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(VECTOR_DB_PATH)


if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents()

    print("Splitting into chunks...")
    chunks = split_documents(docs)

    print("Creating vector database...")
    create_vector_db(chunks)

    print("✅ STEP 3 COMPLETED: Vector database created successfully.")
