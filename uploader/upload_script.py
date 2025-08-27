# uploader/upload_script.py
import time
import faiss
import pandas as pd
import numpy as np
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models
from qdrant_client.models import PayloadSchemaType

# --- 1. FAISS 로드 및 데이터 추출 ---
print("[Uploader] 1. FAISS vector store loading...")
path = "/data/vectorDB/recursive-1000-50_openai-embedding-query-100_faiss"

class DummyEmbeddings(Embeddings):
    def embed_documents(self, texts): raise NotImplementedError
    def embed_query(self, text): raise NotImplementedError

vs = FAISS.load_local(path, DummyEmbeddings(), allow_dangerous_deserialization=True)
index = faiss.read_index(path + "/index.faiss")

extracted_data = []
for pos, doc_id in vs.index_to_docstore_id.items():
    emb = index.reconstruct(pos).astype(np.float32)  # float32 보장
    doc = vs.docstore._dict[doc_id]
    # 기본 스키마 정규화: title(있으면), content(=page_content), date(메타에 있으면), embedding
    meta = dict(doc.metadata) if doc.metadata else {}
    record = {
        "title": meta.get("title"),
        "ellipsis": meta.get("ellipsis"),
        "content": doc.page_content,
        "date": meta.get("date"),  # "YYYY-MM-DD HH:MM" 기대
        "href": meta.get("href"),
        "source_link": meta.get("source_link"),
        "embedding": emb,
        # 필요시 기타 메타 필드 유지
        **{k: v for k, v in meta.items() if k not in {"title", "date"}}
    }
    extracted_data.append(record)

df = pd.DataFrame(extracted_data)
print(f"[Uploader] Success! Converted {len(df)} data points to DataFrame.")

# --- 날짜 문자열 -> 정수 YYYYMMDDHHMM 변환 함수 ---
def to_ymdhm_int(date_str: str):
    if not date_str or not isinstance(date_str, str):
        return None
    # 기본 포맷 고정: "YYYY-MM-DD HH:MM"
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
        return int(dt.strftime("%Y%m%d%H%M"))
    except Exception:
        return None

# 업로드 전에 변환 컬럼 추가
df["date_str"] = df.get("date")
df["date_ymdhm"] = df["date_str"].apply(to_ymdhm_int)

# --- 2. Qdrant 연결 및 컬렉션 생성 ---
print("\n[Uploader] 2. Connecting to Qdrant server...")

client = QdrantClient(host="qdrant", port=6333, timeout=60)

retries = 10
while retries > 0:
    try:
        client.get_collections()
        print("[Uploader] Successfully connected to Qdrant.")
        break
    except Exception as e:
        print(f"[Uploader] Waiting for Qdrant connection... ({retries} retries left) -> {e}")
        retries -= 1
        time.sleep(3)
        if retries == 0:
            raise

collection_name = "faiss_automated_import"
vector_size = len(df.iloc[0]["embedding"])
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
)
print(f"[Uploader] Created collection '{collection_name}' in Qdrant.")

# 날짜 필터 가속 인덱스(정수)
try:
    client.create_payload_index(
        collection_name=collection_name,
        field_name="date_ymdhm",
        field_schema=PayloadSchemaType.INTEGER
    )
except Exception as e:
    print(f"[Uploader] (Info) create_payload_index skipped/failed: {e}")

# --- 3. 데이터 업로드 ---
print("\n[Uploader] 3. Starting data upload...")

def convert_to_native_types(value):
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    return value

points_to_upload = []
for idx, row in df.iterrows():
    # payload 구성: title, content, date_str, date_ymdhm + 기타 메타
    payload_src = row.drop(labels=["embedding"]).to_dict()
    payload = {k: convert_to_native_types(v) for k, v in payload_src.items()}

    # 벡터는 float32 list로
    vec = row["embedding"]
    if isinstance(vec, np.ndarray):
        if vec.dtype != np.float32:
            vec = vec.astype(np.float32)
        vec = vec.tolist()
    else:
        vec = np.asarray(vec, dtype=np.float32).tolist()

    point = models.PointStruct(
        id=idx,                  # id 전략은 필요에 따라 변경(고유키 사용 권장)
        vector=vec,
        payload=payload
    )
    points_to_upload.append(point)

batch_size = 100
total_points = len(points_to_upload)
for i in range(0, total_points, batch_size):
    batch = points_to_upload[i: i + batch_size]
    client.upsert(collection_name=collection_name, wait=True, points=batch)
    print(f"  - Upload progress: {i + len(batch)} / {total_points}")

# --- 4. 최종 확인 ---
collection_info = client.get_collection(collection_name=collection_name)
print(f"\n[Uploader] Upload complete! Collection '{collection_name}' now has {collection_info.points_count} points.")
