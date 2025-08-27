
import os
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
from retriever import load_documents, split_documents, build_or_load_vector_db, create_retriever
from embedding import embedding_documents

from langchain_upstage import ChatUpstage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from qdrant_client import QdrantClient
from typing import List, Dict, Any
from tool.time_line import TimeLine
from qdrant_client.http.models import Distance, VectorParams
load_dotenv()

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    api_key = os.getenv("UPSTAGE_API_KEY")

    filepath = f"./vectorDB/{cfg.split.chunk_type}-{cfg.split.chunk_size}-{cfg.split.chunk_overlap}_{cfg.embedding.backend}-{cfg.embedding.model}-{cfg.embedding.chunk_size}_{cfg.vectordb.backend}"
    #문서 로드
    docs = load_documents(cfg.data.file_path)
    #문서 분할
    documents = split_documents(cfg.split.chunk_type, docs, cfg.split.chunk_size, cfg.split.chunk_overlap)
    #문서 임베딩 모델 로드
    embedding_model = embedding_documents(cfg.embedding.backend, cfg.embedding.model, cfg.embedding.chunk_size, api_key)
    #벡터 데이터베이스 생성
    qdrant_client = None
    if cfg.vectordb.backend == "qdrant":
        qdrant_client = QdrantClient(
            host=cfg.vectordb.client.host, 
            port=cfg.vectordb.client.port,
            timeout=100)
    else:
        build_or_load_vector_db(cfg.vectordb.backend, embedding_model,documents, filepath)
    
    #벡터 데이터베이스 로드 및 검색 설정
    retriever = create_retriever(backend=cfg.vectordb.backend, model=embedding_model, filepath=filepath, client=qdrant_client,collection_name=cfg.vectordb.collection_name)
    #LLM 모델 로드
    llm = ChatUpstage(model=cfg.llm.model, api_key=api_key, temperature=0)
    prompt = PromptTemplate.from_template(
        """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Structure the answer strictly as:
    1) 목차
    2) 내용(핵심 bullet 3~5개, 필요 시 짧은 문단)
    3) 결론/요약(1~2문장)
    If the user asks for timeline/trend explicitly, note it in one sentence and answer with what the context allows.
    Answer in Korean.

    #Question: 
    {question} 
    # Context (each item may come from different sources; do not invent facts):
    {context} 

    #Answer:"""
    )

    def format_docs(docs):
        print("=== 검색된 Context ===")
        formatted_docs = []
        for i, doc in enumerate(docs):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            print(f"[{i}] {doc}\n---")
            formatted_docs.append(content)
        return "\n".join(formatted_docs)

    chain = (
    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )


    #체인 생성
    while True:
        query = input("질문을 입력하세요: ")
        if query == "종료":
            break
                
        response = chain.invoke(query)
        print(response)

if __name__ == "__main__":
    main()
