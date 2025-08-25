from langchain_community.vectorstores import FAISS
import os

def build_or_load_vector_db(backend: str, model, documents: list, filepath: str):
    if backend == "faiss":
        faiss = None
        if not os.path.exists(filepath):
            faiss = FAISS.from_documents(documents, model)
            faiss.save_local(filepath)
    else:
        raise ValueError(f"Invalid backend: {backend}")


def create_retriever(backend: str, model: str, filepath: str):
    if backend == "faiss":
        faiss = FAISS.load_local(filepath, model, allow_dangerous_deserialization=True)
        return faiss.as_retriever(search_type="mmr")
    else:
        raise ValueError(f"Invalid backend: {backend}")