from typing import Any, Dict, List, Optional
from .intent import detect_intent, wants_chronology
from .prompts import make_generic_prompt, make_trend_prompt, make_evolution_prompt
from .chains import (
    build_generic_chain,
    build_trend_chain,
    build_evolution_chain,
    build_timeline_outputs,
)

def make_router(
    retriever,
    llm,
    cfg: Any = None,
    qdrant_client: Any = None,
    collection_name: Optional[str] = None,
    structured_generic: bool = True,
):
    """
    라우터 팩토리.
    반환되는 route()는 다음 시그니처를 가짐:

        route(
            query: str,
            *,
            timeline_button: bool = False,
            timeline_limit: int = 6,
            tags: Optional[List[str]] = None,
        ) -> Dict[str, Any]

    분기 규칙:
    - timeline:  버튼 신호(timeline_button=True)일 때만. 최신순 상위 N개 + 3줄 브리핑 반환.
    - trend:     텍스트에 '트렌드/추이/상승/하락/인기' 등 포함 시.
    - evolution: 텍스트에 '발전 과정/연혁/...' 포함 시 (generic 경로에서 과거→최신).
    - generic:   그 외 전부.
    """

    # generic/trend에서: '발전 과정/연혁'이면 연대기, 아니면 최신우선
    latest_first_fn = lambda q: not wants_chronology(q)

    # 프롬프트 준비
    generic_prompt = make_generic_prompt(structured=structured_generic)
    trend_prompt = make_trend_prompt()
    evo_prompt = make_evolution_prompt()

    # 체인 빌드 (chat_history는 체인 내부에서 []로 주입)
    generic_chain = build_generic_chain(retriever, llm, generic_prompt, latest_first_fn)
    trend_chain = build_trend_chain(retriever, llm, trend_prompt, latest_first_fn)
    evolution_chain = build_evolution_chain(retriever, llm, evo_prompt)

    def route(
        query: str,
        *,
        timeline_button: bool = False,
        timeline_limit: int = 6,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        intent = detect_intent(query, timeline_button=timeline_button)

        if intent == "timeline":
            # 타임라인: 최신순 상위 N개 패널 텍스트 + 채팅창 3줄 브리핑 동시 생성
            out = build_timeline_outputs(
                question=query,
                retriever=retriever,
                llm=llm,
                qdrant_client=qdrant_client,
                collection_name=(
                    collection_name
                    or getattr(getattr(cfg, "vectordb", None), "collection_name", None)
                ),
                limit=timeline_limit,
                tags=tags or [],
                use_llm_postprocess=False,  # 필요 시 True로 후처리
            )
            # 프론트 구분 렌더에 도움 되도록 type 태그를 포함
            out["type"] = "timeline"
            return out

        if intent == "trend":
            # 체인은 질문 문자열만 넘기면 내부에서 context/chat_history를 구성
            text = trend_chain.invoke(query)
            return {"type": "trend", "text": text}

        if wants_chronology(query):
            # 발전 과정/연혁 → generic 경로의 연대기 체인
            text = evolution_chain.invoke(query)
            return {"type": "evolution", "text": text}

        # 기본 generic
        text = generic_chain.invoke(query)
        return {"type": "generic", "text": text}

    return route