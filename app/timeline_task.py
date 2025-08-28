from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from tool.time_line import TimeLine 

class TimelineTask:
    """
    타임라인 전용 태스크.
    - Qdrant 시간 버킷 검색 + Fallback(retriever 최신순)
    - 패널용 타임라인 텍스트 + 3줄 브리핑 생성
    """

    def __init__(
        self,
        *,
        retriever: Any,
        llm: Any,
        embeddings: Any,
        qdrant_client: Any,
        collection_name: str,
        use_llm_postprocess: bool = False,
    ):
        self.retriever = retriever
        self.llm = llm
        self.embeddings = embeddings
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.use_llm_postprocess = use_llm_postprocess

    # 타임스탬프(정수 YYYYMMDDHHMM) 생성 -> time_line.py 스키마에 맞춤
    def _to_ts(self, meta: Dict[str, Any]) -> Optional[int]:
        """
        - 우선순위: date_ymdhm -> published_at -> date -> created_at
        - 정수/정수문자열(YYYYMMDDHHMM or epoch) / ISO 문자열 모두 처리
        """
        def _parse_any(v: Any) -> Optional[int]:
            if v is None:
                return None
            s = str(v).strip()
            if not s:
                return None
            # 숫자면 처리
            if s.isdigit():
                n = int(s)
                # epoch (sec/ms)
                if len(s) in (10, 13):
                    if len(s) == 13:
                        n //= 1000
                    dt = datetime.fromtimestamp(n, tz=timezone.utc)
                    return int(dt.strftime("%Y%m%d%H%M"))
                return int(s[:12].ljust(12, "0"))
            # ISO 시도
            try:
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return int(dt.strftime("%Y%m%d%H%M"))
            except Exception:
                return None

        for key in ("date_ymdhm", "published_at", "date", "created_at"):
            out = _parse_any((meta or {}).get(key))
            if out is not None:
                return out
        return None

    # Qdrant 시간 버킷 검색
    def _search_with_qdrant(
        self,
        question: str,
        *,
        limit: int = 6, # 타임라인 개수 조정 
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
    ) -> List[Dict[str, str]]:
        """
        TimeLine.search_top_per_bucket 로 버킷별 top-1을 뽑고 최신순 상위 N개만 채택.
        """
        items: List[Dict[str, str]] = []
        try:
            # 쿼리 임베딩 (Upstage/여타 임베딩 모델과 호환)
            import numpy as np
            qvec = np.array(self.embeddings.embed_query(question), dtype=np.float32)

            # 기간 기본값(올해 1/1 ~ 지금, UTC)
            now = datetime.now(timezone.utc)
            if start_dt is None:
                start_dt = datetime(now.year, 1, 1) # 파라미터로 기본 기간을 받거나, 최근 N일 옵션으로 설정 가능
            if end_dt is None:
                end_dt = now

            tl = TimeLine(self.qdrant_client, self.collection_name)
            buckets = tl.search_top_per_bucket(
                qvec,
                start_dt=start_dt,
                end_dt=end_dt,
                n_buckets=max(limit, 6),
                limit_per_bucket=1,
            ) or []

            # 최신순 정렬 키: payload.date_ymdhm > bucket_end > bucket_start
            def _key(b):
                p = (b.get("payload") or {})
                return p.get("date_ymdhm") or b.get("bucket_end") or b.get("bucket_start") or 0

            buckets.sort(key=_key, reverse=True)
            buckets = buckets[:max(1, int(limit))]

            for b in buckets:
                p = b.get("payload") or {}
                items.append({
                    "date": p.get("date") or p.get("date_str") or str(b.get("bucket_start") or ""),
                    "title": (p.get("title") or p.get("source") or "이벤트").strip(),
                    "summary": (p.get("content", "") or "").replace("\n", " ").strip(),
                    "source": p.get("site") or p.get("source") or "",
                    "url": p.get("url") or "",
                })
        except Exception:
            # Qdrant 미연결/오류면 빈 리스트로 반환 → 상위에서 fallback
            return []
        return items

    # Retriever fallback (최신순 정렬)
    def _fallback_with_retriever(self, question: str, *, limit: int = 6) -> List[Dict[str, str]]:
        docs = self.retriever.get_relevant_documents(question)
        pairs = []
        for d in docs:
            meta = getattr(d, "metadata", {}) or {}
            ts = self._to_ts(meta)
            pairs.append((ts, d))

        # ts가 None이면 뒤로 가도록, 최신 우선 내림차순
        pairs.sort(key=lambda x: (x[0] is None, x[0]), reverse=True)

        items: List[Dict[str, str]] = []
        for ts, d in pairs[:limit]:
            meta = getattr(d, "metadata", {}) or {}
            items.append({
                "date": meta.get("date") or meta.get("published_at") or meta.get("date_str") or "",
                "title": meta.get("title") or meta.get("source") or "doc",
                "summary": (getattr(d, "page_content", "") or "").replace("\n", " ").strip(),
                "source": meta.get("site") or meta.get("source") or "",
                "url": meta.get("url") or "",
            })
        return items

    def _build_timeline_text(self, items: List[Dict[str, str]], tags: Optional[List[str]] = None) -> str:
        tag_text = ", ".join(tags or [])
        lines = [f"## 타임라인 (태그: {tag_text})"] if tag_text else ["## 타임라인"]
        for it in items:
            src = f" (출처: {it['source']})" if it.get("source") else ""
            url = f"\n  링크: {it['url']}" if it.get("url") else ""
            date = (it.get("date") or "")[:10]
            lines.append(f"- [{date}] {it['title']} — {it['summary'][:160]}{src}{url}")
        txt = "\n".join(lines)

        # 필요 시 LLM 후처리(중복 병합·표현 통일)
        if self.use_llm_postprocess:
            prompt = (
                "당신은 깃뉴스용 타임라인 아나운서입니다.\n"
                "아래 타임라인을 최신순으로 유지하되 중복/유사 항목을 자연스럽게 병합해 주세요.\n"
                "형식은 그대로 유지하세요.\n\n"
                f"{txt}\n"
            )
            try:
                txt = self.llm.invoke(prompt)
            except Exception:
                pass
        return txt

    def _build_briefing(self, timeline_text: str) -> str:
        prompt = (
            "당신은 깃뉴스용 타임라인 아나운서입니다.\n"
            "주어진 타임라인을 근거로 정확히 3줄의 요약 브리핑을 한국어로 작성하세요.\n"
            "각 줄은 1문장 이내, 불릿/링크 없이 텍스트만 출력합니다.\n"
            "1줄: 오늘 기준 핵심 상황, 2줄: 최근 변화, 3줄: 다음 관전 포인트.\n\n"
            f"{timeline_text}\n"
        )
        try:
            return self.llm.invoke(prompt)
        except Exception:
            return "요약 브리핑 생성을 실패했습니다."

    def run(
        self,
        question: str,
        *,
        limit: int = 6,
        tags: Optional[List[str]] = None,
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        반환 형태:
        {
          "type": "timeline",
          "timeline_text": "...",
          "briefing_text": "..."
        }
        """
        items = self._search_with_qdrant(
            question, limit=limit, start_dt=start_dt, end_dt=end_dt
        )

        if not items:
            items = self._fallback_with_retriever(question, limit=limit)

        timeline_text = self._build_timeline_text(items, tags=tags)
        briefing_text = self._build_briefing(timeline_text)

        return {"type": "timeline", "timeline_text": timeline_text, "briefing_text": briefing_text}
