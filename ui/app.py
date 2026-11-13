"""[확장] 금융 PDF 질의응답 화면 (Streamlit). 기존 API 서버를 HTTP로 호출만 한다.

실행(서버를 먼저 띄운다):
    uv run uvicorn --factory src.api.main:build_default_app --port 8000
    uv run --with streamlit streamlit run ui/app.py

- 이 단계에는 대화 메모리와 캐시가 없다. 질문마다 독립 요청이며, 화면에 쌓이는 기록은 보여 주기 위한 것이다.
- 서버 주소는 사이드바에서 바꾸거나 환경 변수 `UI_API_BASE_URL`로 정한다.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 저장소 루트를 import 경로에 추가

import streamlit as st  # noqa: E402

from ui import client  # noqa: E402

STATUS_COLORS = {"answered": "green", "needs_clarification": "orange", "abstained": "gray", "error": "red"}
EXAMPLES = ["루미센서 E는 어떤 제품인가요?", "해온솔루션의 상반기 아시아 매출은 정정 후 얼마인가요?", "해온의 해외 매출을 알려 주세요."]


def render_answer(item: dict) -> None:
    """저장해 둔 답변 하나를 화면에 그린다."""
    r: client.UiAnswer = item["result"]
    color = STATUS_COLORS.get(r.status, "gray")
    st.markdown(f":{color}[**{r.status_label}**]")
    st.write(r.answer)
    if r.sources:
        with st.expander(f"출처 {len(r.sources)}건"):
            for s in r.sources:
                ids = ", ".join(s.source_ids)
                st.markdown(f"- **{s.file_name}** {s.page}쪽" + (f"  ({ids})" if ids else ""))
    if r.reason and not r.is_error:
        with st.expander("답변 판단 이유"):
            st.write(r.reason)
    meta = [f"방식 {client.RAG_METHODS.get(r.rag_method, r.rag_method)}"]
    if r.elapsed_ms is not None:
        meta.append(f"{r.elapsed_ms / 1000:.1f}초")
    if r.error_code:
        meta.append(f"오류 코드 {r.error_code}")
    if r.request_id:
        meta.append(f"요청 {r.request_id[:8]}…")
    st.caption(" · ".join(meta))


def main() -> None:
    st.set_page_config(page_title="금융 PDF 질의응답", page_icon="📄", layout="wide")
    st.title("금융 PDF 질의응답")
    st.caption("PDF에서 근거를 찾아 문서명·쪽과 함께 답합니다. 각 질문은 독립적으로 처리되며 이전 대화를 기억하지 않습니다.")

    if "history" not in st.session_state:
        st.session_state.history = []  # [{"question": str, "result": UiAnswer}]
    if "health" not in st.session_state:
        st.session_state.health = None

    with st.sidebar:
        st.header("설정")
        st.caption("현재 검색 대상: Basic PDF 15개 (FB01~FB15, 360쪽)")
        base_url = st.text_input("API 서버 주소", value=os.environ.get("UI_API_BASE_URL", client.DEFAULT_BASE_URL))
        method = st.selectbox("검색 방식", list(client.RAG_METHODS), format_func=lambda m: client.RAG_METHODS[m])
        if st.button("서버 상태 확인"):
            st.session_state.health = client.check_health(base_url=base_url)
        if st.session_state.health is True:
            st.success("서버가 응답합니다.")
        elif st.session_state.health is False:
            st.error(client.describe_error("server_unreachable"))
        st.divider()
        st.subheader("예시 질문")
        for ex in EXAMPLES:
            st.caption(f"• {ex}")
        if st.button("대화 지우기"):
            st.session_state.history = []
            st.rerun()
        st.divider()
        last = st.session_state.history[-1]["result"] if st.session_state.history else None
        st.caption("마지막 요청")
        st.text(f"request_id: {last.request_id or '-'}\ntrace_id: {last.trace_id or '-'}" if last else "아직 없음")

    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            render_answer(item)

    question = st.chat_input("질문을 입력하세요")
    if question is not None:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("문서를 검색하고 답변을 만드는 중입니다…"):
                result = client.ask(question, method, base_url=base_url)
            item = {"question": question, "result": result}
            render_answer(item)
        st.session_state.history.append(item)
        st.rerun()


main()
