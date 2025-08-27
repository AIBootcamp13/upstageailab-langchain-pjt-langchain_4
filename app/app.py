import os
import streamlit as st
import time
from typing import List, Dict
from datetime import date, timedelta
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ========== Page Config ==========
st.set_page_config(page_title="GeekNews Trend Chat", layout="wide")
st.title("📰 GeekNews Trend Chatbot")

# ========== Sidebar: Controls ==========     왼쪽 태그, 키워드 부분
st.sidebar.header("필터")
ALL_TAGS = ["ai", "ml", "dl", "cv", "llm", "agent", "infra", "data", "tooling"]     # 사용 가능한 태그 리스트
tags: List[str] = st.sidebar.multiselect("태그 선택", ALL_TAGS, default=["ai","llm"])
keyword: str = st.sidebar.text_input("키워드(선택)", placeholder="예: RAG, LangGraph, MLOps")
period = st.sidebar.selectbox("기간", ["최근 7일", "최근 14일", "최근 30일", "직접 선택"])
if period == "직접 선택":
    start = st.sidebar.date_input("시작일", value=date.today() - timedelta(days=14))
    end = st.sidebar.date_input("종료일", value=date.today())
else:
    # '최근 N일' 옵션에 따라 시작일과 종료일을 계산합니다.
    days = int(period.split()[1][:-1])
    start, end = date.today() - timedelta(days=days), date.today()

# '타임라인 생성' 버튼을 생성합니다. 이 버튼을 누르면 do_timeline이 True가 됩니다.
do_timeline = st.sidebar.button("타임라인 생성")

# ========== Session State ==========
# Streamlit의 세션 상태를 사용하여 페이지 리로드 간에 데이터를 유지합니다.
# 'history'가 세션에 없으면 빈 리스트로 초기화하여 대화 기록을 저장합니다.
if "history" not in st.session_state:
    st.session_state.history: List[Dict] = []
if "timeline" not in st.session_state:
    st.session_state.timeline: List[Dict] = []

# ========== API Key & Client Setup ==========
load_dotenv()

try:
    # LangChain을 사용하여 Upstage LLM 클라이언트를 초기화합니다.
    # ChatOpenAI 클래스를 사용하며, Upstage의 API 엔드포인트를 base_url로 지정합니다.
    llm = ChatOpenAI(
        model="solar-1-mini-chat",
        api_key=os.environ["UPSTAGE_API_KEY"],
        base_url="https://api.upstage.ai/v1",
    )
# API 키가 .env 파일에 없을 경우 에러 메시지를 표시하고 앱 실행을 중지합니다.
except (KeyError, TypeError):
    st.error("⚠️ Upstage API 키를 찾을 수 없습니다.")
    st.error("`.env` 파일에 `UPSTAGE_API_KEY = 'YOUR_API_KEY'` 형식으로 키를 저장해주세요.")
    st.stop()

# ========== Mock Backend (교체 포인트) ==========    오른쪽 타임라인 부분  ***** 알고리즘 추가해야함 *****
def fetch_timeline(tags: List[str], keyword: str, start: date, end: date) -> List[Dict]:
    """
    가상 타임라인 데이터를 생성하는 함수입니다.
    TODO: 실제 백엔드 API와 연동하여 데이터를 가져오도록 수정해야 합니다.
    """
    # 백엔드 API 호출 시 발생할 수 있는 지연을 시뮬레이션합니다.
    time.sleep(2)

    base = [{
        "date": str(end - timedelta(days=i*2)),
        "title": f"[{tags[0] if tags else 'trend'}] {keyword or 'hot topic'} 업데이트 {i}",
        "summary": "핵심 포인트 한 줄 요약. (여기는 백엔드 요약 결과를 표시합니다.)",
        "url": "https://news.hada.io/",
        "tags": tags[:3] or ["ai"]
    } for i in range(1,5)]
    return base

# ========== LangChain을 사용하여 Upstage API를 호출하고 챗봇 응답을 생성 ==========
def chat_api(message: str, tags: List[str], keyword: str) -> dict:   
    # 1. 프롬프트 템플릿 정의
    # 시스템 메시지와 사용자 메시지를 포함하는 템플릿을 생성합니다.
    # 'tags', 'keyword', 'question' 변수를 받아 동적으로 프롬프트를 구성합니다.
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant specializing in summarizing and explaining the latest tech trends from sources like GeekNews.
Your answer must be in Korean.
The user is currently interested in the following context:
- Tags: {tags}
- Keyword: {keyword}
Based on this context, please provide a concise and clear answer to the user's question."""),
        ("user", "{question}")
    ])

    # 2. LangChain 체인 구성 (LCEL)
    # 프롬프트, LLM, 출력 파서를 파이프(|)로 연결하여 체인을 만듭니다.
    # StrOutputParser는 LLM의 응답(AIMessage)에서 문자열 내용만 추출합니다.
    chain = prompt_template | llm | StrOutputParser()

    try:
        # 3. 체인 실행
        # 체인에 필요한 변수들을 딕셔너리 형태로 전달하여 실행(invoke)합니다.
        answer = chain.invoke({
            "tags": tags or 'Not specified',
            "keyword": keyword or 'Not specified',
            "question": message
        })
        # UI와 호환되는 딕셔너리 형태로 결과를 반환합니다.
        return {"answer": answer, "citations": []}

    # API 호출 중 에러가 발생하면 Streamlit 화면에 에러 메시지를 표시합니다.
    except Exception as e:
        st.error(f"API 호출 중 오류가 발생했습니다: {e}")
        return {"answer": "죄송합니다, 답변을 생성하는 동안 오류가 발생했습니다.", "citations": []}


# 응답의 근거(Citations)를 화면에 렌더링하는 함수입니다.
def render_citations(citations: List[Dict]):
    # 근거 데이터가 있을 경우에만 확장 가능한(expander) UI를 생성합니다.
    if citations:
        with st.expander("🔍 근거 보기", expanded=False):
            for c in citations:
                st.write(f"- {c['text']}  \n  `{c['source']}`")

# ========== Main Layout ==========
# 메인 화면을 2개의 열(챗봇, 타임라인)으로 분할합니다.  2:1 비율
main_col, right_col = st.columns([2, 1])

# 왼쪽 열 (챗봇 영역)
with main_col:
    st.subheader("💬 챗봇")
    # 높이가 고정된 컨테이너를 만들어, 내용이 넘칠 경우 스크롤이 가능하도록 합니다.
    chat_container = st.container(height=450)
    with chat_container:
        # 세션에 저장된 대화 기록을 순회하며 화면에 렌더링합니다.
        for m in st.session_state.history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                render_citations(m.get("citations", []))

    # 채팅 입력창 위젯을 생성합니다. 사용자가 메시지를 입력하면 user_msg 변수에 저장됩니다.
    user_msg = st.chat_input("메시지를 입력하세요… (예: LLM 트렌드 핵심 요약)")
    if user_msg:
        # 1. 사용자의 메시지를 대화 기록에 추가합니다.
        st.session_state.history.append({"role": "user", "content": user_msg})
        
        # 2. 채팅 컨테이너 내부에 사용자 메시지를 렌더링합니다.
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_msg)

            # 3. 챗봇의 답변을 생성하고 렌더링합니다.
            with st.chat_message("assistant"):
                # '답변 생성 중' 스피너를 표시하여 사용자에게 대기 상태임을 알립니다.
                with st.spinner("답변을 생성하고 있습니다..."):
                    resp = chat_api(user_msg, tags, keyword)
                # API로부터 받은 답변을 렌더링합니다.
                st.markdown(resp["answer"])
                render_citations(resp.get("citations", []))
            
            # 4. 챗봇의 답변을 대화 기록에 추가합니다.
            st.session_state.history.append(
                {"role": "assistant", "content": resp["answer"], "citations": resp.get("citations", [])}
            )

# 오른쪽 열 (타임라인 영역)
with right_col:
    st.subheader("🗓️ 타임라인")
    # '타임라인 생성' 버튼이 눌렸을 때만 타임라인 데이터를 가져옵니다.
    if do_timeline:
        with st.spinner("타임라인을 생성하고 있습니다..."):
            st.session_state.timeline = fetch_timeline(tags, keyword, start, end)
    # 높이가 고정된 컨테이너를 만들어, 타임라인이 길어져도 독립적으로 스크롤되도록 합니다.
    timeline_container = st.container(height=500)
    with timeline_container:
        # 세션에 타임라인 데이터가 있으면 화면에 렌더링합니다.
        if st.session_state.timeline:
            for item in st.session_state.timeline:
                with st.container(border=True): # 각 아이템을 테두리가 있는 컨테이너로 묶습니다.
                    st.markdown(f"**{item['title']}**  \n{item['date']} • {' / '.join(item['tags'])}")
                    st.write(item["summary"])
                    st.markdown(f"[원문 링크]({item['url']})")
        # 데이터가 없으면 안내 메시지를 표시합니다.
        else:
            st.info("좌측 필터에서 **타임라인 생성**을 눌러주세요.")
