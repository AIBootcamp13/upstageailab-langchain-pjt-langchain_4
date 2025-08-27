def wants_chronology(q: str) -> bool:
    """generic 경로에서 사용자가 '발전 과정/연혁'을 요구하는지 텍스트로 감지."""
    low = q.lower()
    ko = ["발전 과정", "변천사", "연혁", "시간순", "흐름", "역사", "처음에", "어떻게 시작"]
    en = ["chronology", "evolution", "history", "progression", "how it started"]
    return any(k in q for k in ko) or any(k in low for k in en)

def detect_intent(q: str, *, timeline_button: bool = False) -> str:
    """
    라우팅 판별 규칙:
    - timeline: 버튼 신호로만 분기 (텍스트 기반 감지 금지)
    - trend: 텍스트로 '트렌드/추이' 등만 감지
    - generic: 그 외 전부 (발전과정/연혁 포함)
    """
    if timeline_button:
        return "timeline"

    low = q.lower()
    if any(k in q for k in ["트렌드", "추이", "상승", "하락", "인기"]) or "trend" in low:
        return "trend"

    # 발전 과정/연혁은 generic 경로에서 wants_chronology로 후처리
    return "generic"