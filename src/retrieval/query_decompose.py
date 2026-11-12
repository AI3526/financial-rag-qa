"""질문 분해(다중 질의) — 확장 브랜치 5a (EXTENSION_INTERMEDIATE_LOG).

모델이 질문을 최대 3개의 검색 질의로 나눈다. 프롬프트는 `prompts/query_decompose_v1.json`(사용자 승인 문안)이다.
- 입력은 질문 문자열뿐이다. 평가 정답·근거는 이 모듈에 들어오지 않는다.
- 결과 질의에서 원래 질문과 같은 것·중복은 빼고 최대 3개로 자른다. 검색 때는 원래 질문을 항상 함께 쓴다(`multi_query.py`).
- 모델 출력이 잘못되었거나 호출이 실패하면 추가 질의 없이 원래 질문만 쓰도록 빈 목록을 돌려주고 실패 사유를 기록한다(검색을 막지 않는다).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = ROOT / "prompts" / "query_decompose_v1.json"
MAX_QUERIES = 3
RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "search_queries", "strict": True, "schema": {
    "type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"}}}, "required": ["queries"], "additionalProperties": False}}}


def load_prompt(path: Path = PROMPT_PATH) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    spec["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return spec


def build_messages(spec: dict, question: str) -> list[tuple[str, str]]:
    return [("system", spec["system"]), ("human", spec["user_template"].format(question=question))]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def parse_queries(text: str | None, original: str, limit: int = MAX_QUERIES) -> tuple[list[str], str | None]:
    """(추가 검색 질의 목록, 실패 사유). 원래 질문과 같은 질의·빈 질의·중복은 버리고 최대 limit개로 자른다."""
    try:
        data = json.loads(text or "")
        raw = data["queries"]
        if not isinstance(raw, list) or not all(isinstance(q, str) for q in raw):
            return [], "queries가 문자열 목록이 아님"
    except (ValueError, KeyError, TypeError) as exc:
        return [], f"출력을 해석하지 못함({type(exc).__name__})"
    seen, out = {_norm(original)}, []
    for q in raw:
        n = _norm(q)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out[:limit], None if raw else "queries가 비어 있음"


def build_llm(*, timeout: float = 60, max_completion_tokens: int = 2048, reasoning_effort: str = "low", seed: int = 20261111):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=os.environ["OPENAI_MODEL"], max_retries=2, timeout=timeout, reasoning_effort=reasoning_effort,
                      max_completion_tokens=max_completion_tokens, seed=seed, model_kwargs={"response_format": RESPONSE_FORMAT})


def decompose(llm, spec: dict, question: str) -> dict:
    """질문 1개를 분해한다. 예외는 던지지 않고 `error`에 담는다."""
    from src.generation.generator import _text, _usage

    started = time.perf_counter()
    row = {"question": question, "queries": [], "raw": None, "usage": None, "error": None}
    try:
        msg = llm.invoke(build_messages(spec, question))
        row["raw"], row["usage"] = _text(msg.content), _usage(msg)
        row["queries"], row["error"] = parse_queries(row["raw"], question, int(spec.get("max_queries", MAX_QUERIES)))
    except Exception as exc:  # noqa: BLE001 - 호출 실패가 평가 전체를 멈추지 않게 기록한다
        row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    row["seconds"] = round(time.perf_counter() - started, 2)
    return row
