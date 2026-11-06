import os

from src import config


def test_empty_cache_variables_are_treated_as_unset(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("HF_HOME=\nQDRANT_PATH= \nSOMETHING_ELSE=\nKEEP=value\n", encoding="utf-8")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    for name in ("HF_HOME", "QDRANT_PATH", "SOMETHING_ELSE", "KEEP"):
        monkeypatch.delenv(name, raising=False)
    config.load_env()
    assert "HF_HOME" not in os.environ and "QDRANT_PATH" not in os.environ  # 빈 값은 제거
    assert os.environ["KEEP"] == "value"  # 값이 있으면 유지
    assert os.environ["SOMETHING_ELSE"] == ""  # 목록 밖 변수는 건드리지 않는다
    monkeypatch.delenv("SOMETHING_ELSE", raising=False)
    monkeypatch.delenv("KEEP", raising=False)
