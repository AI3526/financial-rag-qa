from src.observability import langfuse_client as lc


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
