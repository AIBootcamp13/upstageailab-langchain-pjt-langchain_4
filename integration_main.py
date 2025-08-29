

from tools.time_line import TimeLine
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
from retriever import load_documents, split_documents, build_or_load_vector_db, create_retriever
from embedding import embedding_documents
import os
from langchain_upstage import ChatUpstage
from qdrant_client import QdrantClient
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from datetime import datetime, timedelta
import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA
load_dotenv()
system_prompt = """
   당신은 사용자의 질문을 분석하여 두 가지 작업을 수행하는 전문가입니다.
    
    1.  **IT 기술 용어 추출**: 질문에서 언급된 IT 관련 기술적 고유 명사(예: 프로그래밍 언어, 프레임워크, 라이브러리, 데이터베이스, 클라우드 서비스, 프로토콜, 운영체제 등)를 모두 추출합니다.
    2.  **타임라인 질문 판별**: 질문이 특정 기간, 일정, 계획 등 시간과 관련된 정보를 요구하는지 판별합니다. "언제까지", "얼마나 걸릴까?", "일정", "계획", "로드맵" 등의 키워드가 포함되면 타임라인 질문으로 간주합니다.

    분석 결과를 반드시 다음 JSON 형식으로만 출력해야 합니다.
    {
      "it_terms": ["용어1", "용어2", ...],
      "is_timeline_query": true/false
    }

    - `it_terms`: 추출된 IT 기술 용어 배열. 없으면 빈 배열 `[]`로 설정합니다.
    - `is_timeline_query`: 타임라인 관련 질문이면 `true`, 아니면 `false`로 설정합니다.
    - 설명이나 다른 문장은 절대 추가하지 말고, 오직 지정된 JSON 객체만 출력해야 합니다.
"""
class Main:
    def __init__(self,llm,embedding_model,qdrant_client,retriever):
        self.llm = llm
        self.embedding_model = embedding_model
        self.qdrant_client = qdrant_client
        self.retriever = retriever
        self.timeline = TimeLine(self.qdrant_client)
        self.qa_retriever = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.retriever)

    def rount(self,msg:str,tags=[]):
        messages = [
            SystemMessage(
                content=f"""{system_prompt}"""
            ),
            HumanMessage(
                content=f"""[HumanMessage]: {msg} \n [Json]: """
            )
        ]
        parser = JsonOutputParser()
        response = self.llm.invoke(messages)
        response = parser.parse(response.content)
        print(response)
        return dict(tags=tags,keywords=response['it_terms'],timeline=response['is_timeline_query'])

    def to_embedding(self,text):
        embedding_response = self.embedding_model.embed_query(text)
        return np.array(embedding_response,dtype=np.float32)



    def run(self, tags=[], msg='',start_dt=datetime(2025, 1, 1, 0, 0), end_dt=datetime(2025, 8, 27, 23, 59), n_buckets=10): 
        rount_response = self.rount(msg,tags)
        timeline_vaild = rount_response['timeline']
        keywords = rount_response['keywords']
        if tags+keywords:
            tags_keywords = ' '.join(tags+keywords)
            if timeline_vaild:
                query_vector = self.to_embedding(tags_keywords)
                timeline_response = self.timeline.search_top_per_bucket(query_vector,start_dt, end_dt, n_buckets=n_buckets)


                messages = self.timeline.timeline_prompt(timeline_response)
                response = self.llm.invoke(messages)
                response = response.content
                return (response,timeline_response)
            else:
                result = self.qa_retriever.invoke({"query": tags_keywords})
                return result['result']
                

        return self.llm.invoke(
            [
                HumanMessage(
                    content=msg
                )
            ]
        ).content

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    api_key = os.getenv("UPSTAGE_API_KEY")

    filepath = f"./vectorDB/{cfg.split.chunk_type}-{cfg.split.chunk_size}-{cfg.split.chunk_overlap}_{cfg.embedding.backend}-{cfg.embedding.model}-{cfg.embedding.chunk_size}_{cfg.vectordb.backend}"

    #문서 임베딩 모델 로드
    embedding_model = embedding_documents(cfg.embedding.backend, cfg.embedding.model, cfg.embedding.chunk_size, api_key)
    #벡터 데이터베이스 생성
    qdrant_client = QdrantClient(
        host=cfg.vectordb.client.host, 
        port=cfg.vectordb.client.port,
        timeout=100)
    
    #벡터 데이터베이스 로드 및 검색 설정
    retriever = create_retriever(backend=cfg.vectordb.backend, model=embedding_model, filepath=filepath, client=qdrant_client,collection_name=cfg.vectordb.collection_name)
    #LLM 모델 로드
    llm = ChatUpstage(model=cfg.llm.model, api_key=api_key, temperature=0)
    main = Main(llm,embedding_model,qdrant_client,retriever)
    print(main.run(msg=""))

if __name__ == "__main__":
    main()