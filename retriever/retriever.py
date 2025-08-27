from langchain_community.vectorstores import FAISS
from langchain_qdrant import Qdrant
from typing import Any
import os

def build_or_load_vector_db(backend: str, model, documents: list, filepath: str):
    if backend == "faiss":
        faiss = None
        if not os.path.exists(filepath):
            faiss = FAISS.from_documents(documents, model)
            faiss.save_local(filepath)
    if backend == "qdrant":
        pass
    else:
        raise ValueError(f"Invalid backend: {backend}")


def create_retriever(backend: str, model: str, filepath: str,**kwargs: Any):
    if backend == "faiss":
        faiss = FAISS.load_local(filepath, model, allow_dangerous_deserialization=True)
        return faiss.as_retriever(search_type="mmr")
    
    if backend == "qdrant":
        from langchain_qdrant import QdrantVectorStore
        if 'client' not in kwargs or 'collection_name' not in kwargs:
            raise ValueError("Arguments 'client' and 'collection_name' are required for the 'qdrant' backend.")

        qdrant_vector_store = Qdrant(
            client=kwargs['client'],
            collection_name=kwargs['collection_name'],
            embeddings=model,
            content_payload_key='content'
        )
        return qdrant_vector_store.as_retriever(
            search_type="mmr", 
            search_kwargs={'k': 5}
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")