from langchain_upstage import UpstageEmbeddings


def embedding_documents(backend: str, model: str, chunk_size: int, api_key: str):
    if backend == "openai":
        return UpstageEmbeddings(model=model,api_key=api_key,chunk_size=chunk_size)
    else:
        raise ValueError(f"Invalid backend: {backend}")