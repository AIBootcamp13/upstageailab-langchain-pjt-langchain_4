from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter

def split_documents(chunk_type: str,docs: list, chunk_size: int, chunk_overlap: int):
    if chunk_type == "recursive":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    elif chunk_type == "character":
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif chunk_type == "token":
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        raise ValueError(f"Invalid chunk type: {chunk_type}")
    return text_splitter.split_documents(docs)