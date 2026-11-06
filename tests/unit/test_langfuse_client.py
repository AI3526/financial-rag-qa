import re

import pytest

from src.observability import langfuse_client as lc


def test_trace_id_is_deterministic_32_hex():
    a = lc.trace_id_for("req-1")
    b = lc.trace_id_for("req-1")
    c = lc.trace_id_for("req-2")
    assert a == b
    assert a != c
    assert re.fullmatch(r"[0-9a-f]{32}", a)


def test_new_request_ids_generates_unique_request_ids():
    one = lc.new_request_ids()
    two = lc.new_request_ids()
    assert one.request_id != two.request_id
    assert one.trace_id == lc.trace_id_for(one.request_id)


def test_new_request_ids_accepts_given_request_id():
    ids = lc.new_request_ids("fixed-id")
    assert ids.request_id == "fixed-id"
    assert ids.trace_id == lc.trace_id_for("fixed-id")


def test_get_langfuse_returns_none_without_keys(monkeypatch):
    # .env를 다시 읽지 않도록 load_dotenv를 막고 변수를 제거한다
    monkeypatch.setattr(lc, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert lc.langfuse_configured() is False
    assert lc.get_langfuse() is None


def test_get_langfuse_swallows_init_errors(monkeypatch):
    monkeypatch.setattr(lc, "langfuse_configured", lambda: True)

    def boom():
        raise RuntimeError("secret-should-not-leak")

    import langfuse

    monkeypatch.setattr(langfuse, "get_client", boom)
    assert lc.get_langfuse() is None  # 예외가 밖으로 나오지 않는다


def test_secret_values_never_appear_in_log_on_failure(monkeypatch, caplog):
    monkeypatch.setattr(lc, "langfuse_configured", lambda: True)

    def boom():
        raise RuntimeError("sk-lf-SECRETVALUE")

    import langfuse

    monkeypatch.setattr(langfuse, "get_client", boom)
    with caplog.at_level("WARNING"):
        lc.get_langfuse()
    assert "SECRETVALUE" not in caplog.text
