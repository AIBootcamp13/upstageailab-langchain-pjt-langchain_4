import os
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
from retriever import load_documents, split_documents, build_or_load_vector_db, create_retriever
from embedding import embedding_documents
import numpy as np
from datetime import datetime, timedelta

from langchain_upstage import ChatUpstage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from qdrant_client import QdrantClient
from typing import List, Dict, Any
from tool.time_line import TimeLine
from qdrant_client.http.models import Distance, VectorParams
from model import make_router
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
    # ─────────────────────────────────────────────
# (중요) 타임라인 경로에서 질의 임베딩이 필요합니다.
# retriever에 embedder가 없다면, 기존 embedding_model을 붙여 주세요.
# ─────────────────────────────────────────────
try:
    _ = retriever.embeddings  # 존재 확인
except AttributeError:
    if 'embedding_model' in locals():
        setattr(retriever, "embeddings", embedding_model)  # 간단히 주입
    else:
        # 임베딩이 전혀 없다면 타임라인 기능은 폴백으로만 동작합니다.
        pass

# ─────────────────────────────────────────────
# Qdrant 클라이언트가 있으면 넘기고, 없으면 None으로 둡니다.
# ─────────────────────────────────────────────
qdrant = qdrant_client if 'qdrant_client' in locals() else None
collection = getattr(getattr(cfg, "vectordb", None), "collection_name", None)

# ─────────────────────────────────────────────
# 라우터 생성 (메인은 얇게, 로직은 model/ 쪽으로)
# ─────────────────────────────────────────────
router = make_router(
    retriever=retriever,
    llm=llm,
    cfg=cfg,
    qdrant_client=qdrant,
    collection_name=collection,
    structured_generic=True,  # 일반 QA는 목차/내용/결론 구조 사용
)

# ─────────────────────────────────────────────
# 사용 예시 ①: 일반 채팅 질의
# ─────────────────────────────────────────────
def answer_chat(query: str):
    """
    일반 입력을 라우터에 넘기고, 반환 타입에 따라 출력 채널 분배.
    UI가 있다면 아래 print 대신 패널/채팅창으로 분기 렌더링하세요.
    """
    result = router(query)  # timeline_button=False 기본
    rtype = result.get("type")

    if rtype == "timeline":
        # 이 경로는 보통 버튼으로만 들어옵니다. (여기선 방어코드)
        print("=== [타임라인 패널] ===")
        print(result["timeline_text"])
        print("\n=== [채팅창 브리핑(3줄)] ===")
        print(result["briefing_text"])
    else:
        # generic / trend / evolution 공통
        print(result.get("text", ""))

# ─────────────────────────────────────────────
# 사용 예시 ②: 타임라인 버튼 클릭 시
# ─────────────────────────────────────────────
def answer_timeline(query: str, limit: int = 6, tags=None):
    """
    UI에서 타임라인 버튼을 눌렀을 때 호출.
    반환 객체에 패널용/채팅창용이 분리되어 들어옵니다.
    """
    result = router(
        query,
        timeline_button=True,      # ★ 버튼 신호로만 타임라인 분기
        timeline_limit=limit,
        tags=tags or [],
    )
    # === 여기서 UI에 분리 렌더 ===
    timeline_panel_text = result["timeline_text"]  # 타임라인 노출 칸
    briefing_3lines     = result["briefing_text"]  # 채팅창

    # 데모 출력
    print("=== [타임라인 패널] ===")
    print(timeline_panel_text)
    print("\n=== [채팅창 브리핑(3줄)] ===")
    print(briefing_3lines)

# ─────────────────────────────────────────────
# (옵션) 간단 REPL 데모
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("입력 예시: 일반 질문 그대로 /timeline 주제 [개수]  형태")
    while True:
        q = input("> ").strip()
        if q in ("종료", "exit", "quit"):
            break
        if q.startswith("/timeline"):
            parts = q.split(maxsplit=2)
            lim = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 6
            topic = parts[1] if len(parts) >= 2 else ""
            answer_timeline(topic or "타임라인", limit=lim, tags=[topic] if topic else [])
        else:
            answer_chat(q)