# search.py
import numpy as np
from qdrant_client import QdrantClient,models
from datetime import datetime, timedelta
import math
from qdrant_client.models import Filter, FieldCondition, Range
from qdrant_client import QdrantClient
from openai import OpenAI
from dotenv import load_dotenv
import os

class TimeLine:
    def __init__(self, qdrant_client, COLLECTION_NAME: str="faiss_automated_import",QDRANT_HOST: str="localhost",QDRANT_PORT: int=50333):
        self.qdrant_client = qdrant_client
        self.COLLECTION_NAME = COLLECTION_NAME
    def to_ymdhm_int(self, dt: datetime) -> int:

        return int(dt.strftime("%Y%m%d%H%M"))
    
    def search_top_per_bucket(self, query_vector, start_dt: datetime, end_dt: datetime, n_buckets: int = 10, limit_per_bucket: int = 1):
        if end_dt <= start_dt:
            raise ValueError("end_dt는 start_dt보다 커야 합니다.")
        
        # 전체 구간을 동일한 '분' 단위로 등분
        total_minutes = int((end_dt - start_dt).total_seconds() // 60)
        # 버킷 크기(분). 끝 경계 포함(lte)로 잡을 것이므로 -1분 여유를 둠
        bucket_minutes = math.ceil((total_minutes + 1) / n_buckets)

        results = []
        seen_ids = set()

        for i in range(n_buckets):
            b_start = start_dt + timedelta(minutes=i * bucket_minutes)
            b_end   = min(end_dt, b_start + timedelta(minutes=bucket_minutes - 1))
            if b_start > end_dt:
                break

            f = Filter(
                must=[
                    FieldCondition(
                        key="date_ymdhm",
                        range=Range(gte=self.to_ymdhm_int(b_start), lte=self.to_ymdhm_int(b_end))
                    )
                ]
            )

            hits = self.qdrant_client.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=query_vector,
                limit=limit_per_bucket,
                query_filter=f,
                with_payload=True
            )

            # 해당 버킷에 결과가 없을 수 있음
            for h in hits:
                # 버킷 간 중복 방지(같은 포인트가 여러 버킷에 걸릴 수 있으므로)
                if h.id in seen_ids:
                    continue
                seen_ids.add(h.id)
                results.append({
                    "id": h.id,
                    "score": h.score,
                    "payload": h.payload,
                    "bucket_index": i,
                    "bucket_start": b_start,
                    "bucket_end": b_end,
                })

        # 반환 순서: 버킷 순(시간순). 필요하면 score로 정렬 변경 가능
        results.sort(key=lambda x: x["bucket_index"])
        return results


if __name__ == "__main__":
    load_dotenv()
    api_key=os.environ["UPSTAGE_API_KEY"]

    from openai import OpenAI
    
    embeding_model = OpenAI(
        api_key=api_key,
        base_url="https://api.upstage.ai/v1"
    )
 
    response = embeding_model.embeddings.create(
        input="android ai",
        model="embedding-query"
    )

    query_vector = np.array(response.data[0].embedding,dtype=np.float32)


    QDRANT_HOST = "localhost"
    QDRANT_PORT = 50333
    COLLECTION_NAME = "faiss_automated_import"
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10000)
    timeline = TimeLine(qdrant_client, COLLECTION_NAME)
    
    start_dt = datetime(2025, 1, 1, 0, 0)
    end_dt   = datetime(2025, 1, 27, 23, 59)
    top10 = timeline.search_top_per_bucket(query_vector, start_dt, end_dt, n_buckets=10)
    for r in top10:
        print(r["bucket_index"], r["id"], r["score"], r["payload"])