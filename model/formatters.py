from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

def create_ts(meta: Dict[str, Any]):
    """정렬용 분 단위 타임스탬프(YYYYMMDDHHMM) 생성.
    우선순위: date_ymdhm → published_at → date → created_at
    숫자/숫자문자열(YYYYMMDDHHMM or epoch), ISO 문자열 모두 처리."""
    for k in ("date_ymdhm", "published_at", "date", "created_at"): # DB 스키마에 맞게 수정
        v = (meta or {}).get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s.isdigit():
            n = int(s)
            if len(s) in (10, 13):  # epoch (sec/ms)
                if len(s) == 13:
                    n //= 1000
                dt = datetime.fromtimestamp(n, tz=timezone.utc)
                return int(dt.strftime("%Y%m%d%H%M"))
            return int(s[:12].ljust(12, "0"))
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.strftime("%Y%m%d%H%M"))
        except Exception:
            continue
    return None

def format_docs(docs: List[Any], latest_first: bool = True, per_doc_limit: int = 1200) -> str:
    """리트리버 문서를 정렬·병합해 LLM 컨텍스트 문자열로 변환."""
    sortable: List[Tuple[int, Any]] = [(create_ts(getattr(d, "metadata", {}) or {}), d) for d in docs]
    sortable.sort(key=lambda x: (x[0] is None, x[0]), reverse=bool(latest_first)) # ts가 None인 건 뒤로. latest_first=True면 내림차순(최신→과거)

    print("=== 검색된 Context(시간 정렬) ===")

    parts: List[str] = []
    for i, (ts, d) in enumerate(sortable, start=1):
        meta = getattr(d, "metadata", {}) or {}
        title = meta.get("title") or meta.get("source") or "doc"
        date_label = meta.get("published_at") or meta.get("date") or meta.get("created_at") or meta.get("date_ymdhm")
        content = getattr(d, "page_content", str(d)) or ""
        header = f"[{i}] {title}" + (f" | {date_label}" if date_label else "")  # llm 전달 형태 (header) : [1] GitNews | 2025-08-25
        snippet = content.strip().replace("\n", " ")[:per_doc_limit]
        parts.append(f"{header}\n{snippet}")
    return "\n---\n".join(parts)