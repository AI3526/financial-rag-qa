"""[확장] Streamlit 화면 테스트: 서버 호출(client.ask)을 가짜로 바꿔 화면 동작만 확인한다.

streamlit은 프로젝트 필수 의존성이 아니므로 없으면 건너뛴다(`uv run --with streamlit pytest tests/unit/test_ui_app.py`).
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from ui import client  # noqa: E402

APP = str(Path(__file__).resolve().parents[2] / "ui" / "app.py")


def _result(status="answered", **over):
    base = dict(status=status, answer="답변 본문", reason="판단 이유", rag_method="hybrid_rerank",
                sources=[client.SourceRef("FB05", "FB05.pdf", 15, ("S1", "S2"))], request_id="abcdef1234567890",
                trace_id="t" * 32, http_status=200, elapsed_ms=2500)
    base.update(over)
    return client.UiAnswer(**base)


def _text(at):
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [t.value for t in at.text]
    return "\n".join(str(p) for p in parts)


def test_처음_화면에는_안내와_입력창이_있다():
    at = AppTest.from_file(APP).run()
    assert not at.exception
    assert at.title[0].value == "금융 PDF 질의응답"
    assert "이전 대화를 기억하지 않습니다" in _text(at)
    assert "현재 검색 대상: Basic PDF 15개" in _text(at)  # 검색 대상 안내
    assert len(at.chat_input) == 1 and at.selectbox[0].value == "hybrid_rerank"  # 기본 방식은 선택된 방식


def test_질문하면_답변과_출처와_요청_정보를_보여_준다(monkeypatch):
    calls = []
    monkeypatch.setattr(client, "ask", lambda q, m, **kw: calls.append((q, m, kw.get("base_url"))) or _result())
    at = AppTest.from_file(APP).run()
    at.chat_input[0].set_value("루미센서 E는 무엇인가요?").run()
    assert not at.exception
    assert calls == [("루미센서 E는 무엇인가요?", "hybrid_rerank", client.DEFAULT_BASE_URL)]
    text = _text(at)
    assert "답변 본문" in text and "FB05.pdf" in text and "15쪽" in text and "abcdef12" in text
    assert "request_id: abcdef1234567890" in text  # 사이드바의 마지막 요청


def test_선택한_검색_방식으로_호출한다(monkeypatch):
    calls = []
    monkeypatch.setattr(client, "ask", lambda q, m, **kw: calls.append(m) or _result(rag_method=m))
    at = AppTest.from_file(APP).run()
    at.selectbox[0].set_value("dense").run()
    at.chat_input[0].set_value("질문").run()
    assert calls == ["dense"]


def test_오류는_안내_문구와_오류_코드를_보여_준다(monkeypatch):
    monkeypatch.setattr(client, "ask", lambda q, m, **kw: client.UiAnswer(
        status="error", answer=client.describe_error("model_timeout"), reason="model_timeout", rag_method=m,
        error_code="model_timeout", http_status=504, request_id="r" * 16, elapsed_ms=31000))
    at = AppTest.from_file(APP).run()
    at.chat_input[0].set_value("질문").run()
    assert not at.exception
    text = _text(at)
    assert client.describe_error("model_timeout") in text and "오류 코드 model_timeout" in text


def test_대화_지우기는_기록을_비운다(monkeypatch):
    monkeypatch.setattr(client, "ask", lambda q, m, **kw: _result())
    at = AppTest.from_file(APP).run()
    at.chat_input[0].set_value("질문").run()
    assert len(at.chat_message) == 2
    next(b for b in at.button if b.label == "대화 지우기").click().run()
    assert len(at.chat_message) == 0
