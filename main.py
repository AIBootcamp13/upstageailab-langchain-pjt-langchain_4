
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
from typing import List, Dict, Any
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
    build_or_load_vector_db(cfg.vectordb.backend, embedding_model,documents, filepath)
    #벡터 데이터베이스 로드 및 검색 설정
    retriever = create_retriever(cfg.vectordb.backend, embedding_model, filepath)
    #LLM 모델 로드
    llm = ChatUpstage(model=cfg.llm.model, api_key=api_key, temperature=0)
    prompt = PromptTemplate.from_template(
        """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Answer in Korean.

    #Question: 
    {question} 
    #Context: 
    {context} 

    #Answer:"""
    )

    def format_docs(docs):
        print("=== 검색된 Context ===")
        for i, doc in enumerate(docs):
            print(f"[{i}] {doc.page_content}\n---")
        return docs

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
