
import os
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
from retriever import load_documents, split_documents, build_or_load_vector_db, create_retriever
from embedding import embedding_documents

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
    
    print(retriever.get_relevant_documents("AI 최신 기술 검색"))


if __name__ == "__main__":
    main()
