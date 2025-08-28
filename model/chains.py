from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from .prompts import make_timeline_prompt, make_timeline_briefing_prompt
from typing import Dict, Any, List



def build_generic_chain(retriever, llm, prompt, latest_first_fn):
    def ctx_builder(question: str):
        docs = retriever.get_relevant_documents(question)
        from .formatters import format_docs
        return format_docs(docs, latest_first=latest_first_fn(question))
    return (
        {
            "context": RunnableLambda(ctx_builder),
            "question": RunnablePassthrough(),
            "chat_history": RunnableLambda(lambda _: []) 
        }
        | prompt | llm | StrOutputParser()
    )

def build_trend_chain(retriever, llm, trend_prompt, latest_first_fn):
    def ctx_builder(question: str):
        docs = retriever.get_relevant_documents(question)
        from .formatters import format_docs
        return format_docs(docs, latest_first=latest_first_fn(question))
    return (
        {
            "context": RunnableLambda(ctx_builder),
            "question": RunnablePassthrough(),
            "chat_history": RunnableLambda(lambda _: []) 
        }
        | trend_prompt | llm | StrOutputParser()
    )

def build_evolution_chain(retriever, llm, evo_prompt):
    def ctx_builder(question: str):
        docs = retriever.get_relevant_documents(question)
        from .formatters import format_docs
        return format_docs(docs, latest_first=False) 
    return (
        {
            "context": RunnableLambda(ctx_builder),
            "question": RunnablePassthrough(),
            "chat_history": RunnableLambda(lambda _: []) 
        }
        | evo_prompt | llm | StrOutputParser()
    )

# 타임라인: 패널용 타임라인 + 채팅창 브리핑을 동시에 생성
def build_timeline_outputs(
    question: str,
    retriever,
    llm,
    *, # 인자 -> 키워드 전용으로 받음
    qdrant_client: Any = None,
    collection_name: str | None = None,
    limit: int = 6, # 타임라인 : 최신 6개 반환
    tags: List[str] | None = None,
    use_llm_postprocess: bool = False, # True : Timeline list 추가 정제(중복 병합/표현 통일)
) -> Dict[str, str]:
    """
    반환:
      {
        "timeline_text": "...패널에 뿌릴 타임라인 리스트...",
        "briefing_text": "...채팅창에 뿌릴 3줄 브리핑..."
      }
    """
    # 1) 버킷 검색 (없으면 RAG로 폴백 리스트)
    items: List[Dict[str, str]] = []

    if qdrant_client and collection_name:
        try:
            from tool.time_line import TimeLine
            import numpy as np
            qvec = np.array(retriever.embeddings.embed_query(question), dtype=np.float32)
            tl = TimeLine(qdrant_client, collection_name)
            buckets = tl.search_top_per_bucket(qvec) or []
            # 최신순 + 상위 N
            def _key(b):
                p = (b.get("payload") or {})
                return p.get("date_ymdhm") or b.get("bucket_end") or b.get("bucket_start") or 0
            buckets.sort(key=_key, reverse=True)
            buckets = buckets[:max(1, int(limit))]
            for b in buckets:
                p = b.get("payload") or {}
                items.append({
                    "date": p.get("date") or p.get("date_str") or str(b.get("bucket_start")),
                    "title": (p.get("title") or p.get("source") or "이벤트").strip(),
                    "summary": (p.get("content", "") or "").replace("\n", " ").strip(),
                    "source": p.get("site") or p.get("source") or "",
                    "url": p.get("url") or "",
                })
        except Exception:
            pass

    if not items:
        # 폴백: 리트리버 top-k를 최신순 정렬 후 N개만
        docs = retriever.get_relevant_documents(question)
        from .formatters import create_ts
        # 간단히 최신순으로 자르기
        scored = [(create_ts(getattr(d, "metadata", {}) or {}), d) for d in docs]
        scored.sort(key=lambda x: (x[0] is None, x[0]), reverse=True)
        for ts, d in scored[:limit]:
            meta = getattr(d, "metadata", {}) or {}
            items.append({
                "date": meta.get("date") or meta.get("published_at") or meta.get("date_str") or "",
                "title": meta.get("title") or meta.get("source") or "doc",
                "summary": (getattr(d, "page_content", "") or "").replace("\n", " ").strip(),
                "source": meta.get("site") or meta.get("source") or "",
                "url": meta.get("url") or "",
            })

    # 2) 패널용 타임라인 텍스트 (코드로 먼저 생성)
    tag_text = ", ".join(tags or [])
    lines = [f"## 타임라인 (태그: {tag_text})"] if tag_text else ["## 타임라인"]
    for it in items:
        src = f" (출처: {it['source']})" if it.get("source") else ""
        url = f"\n  링크: {it['url']}" if it.get("url") else ""
        date = (it.get("date") or "")[:10]
        lines.append(f"- [{date}] {it['title']} — {it['summary'][:160]}{src}{url}")
    timeline_text = "\n".join(lines)

    # 3) LLM 후처리로 타임라인 텍스트 다듬기
    if use_llm_postprocess:
        tl_prompt = make_timeline_prompt()
        tl_msgs = tl_prompt.format_messages(
            tags=tag_text, context=timeline_text, max_items=len(items), chat_history=[]
        )
        timeline_text = llm.invoke(tl_msgs)

    # 4) 채팅창용 3줄 브리핑 생성
    brief_prompt = make_timeline_briefing_prompt()
    brief_msgs = brief_prompt.format_messages(timeline_text=timeline_text, chat_history=[])
    briefing_text = llm.invoke(brief_msgs)

    return {"timeline_text": timeline_text, "briefing_text": briefing_text}
