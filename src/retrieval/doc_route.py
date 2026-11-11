"""문서 단위 라우팅 — 확장 브랜치 방법 A (EXTENSION_INTERMEDIATE_LOG).

모델이 문서 목록(번호·제목)과 질문을 보고 읽어야 할 문서를 최대 4개 고른다. 프롬프트는 `prompts/doc_route_v1.json`(사용자 승인 문안)이다.
- 문서 목록은 `pages.jsonl`의 각 문서 첫 쪽에서 문서 번호와 문서 제목을 읽어 만든다(PDF 안의 제목이며 평가 정보와 무관하다).
- 입력은 질문 문자열과 문서 목록뿐이다. 평가 정답·근거는 이 모듈에 들어오지 않는다.
- 모델이 목록에 없는 번호를 만들면 버린다(개수는 기록). 출력이 잘못되었거나 호출이 실패하면 빈 목록(라우팅 없음)을 돌려주고 사유를 기록한다.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

from src.ingest import chunk as ck

ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = ROOT / "prompts" / "doc_route_v1.json"
MAX_DOCS = 4
RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "doc_ids", "strict": True, "schema": {
    "type": "object", "properties": {"doc_ids": {"type": "array", "items": {"type": "string"}}}, "required": ["doc_ids"], "additionalProperties": False}}}


def load_prompt(path: Path = PROMPT_PATH) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    spec["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return spec


def build_catalog(pages: list[dict]) -> list[tuple[str, str]]:
    """(문서 번호, 문서 제목) 목록. 각 문서의 첫 쪽에서 제목을 읽는다. 첫 쪽 구조가 예상과 다르면 제목 없이 번호만 남긴다."""
    first: dict[str, dict] = {}
    for p in pages:
        if p["book_id"] not in first or p["page"] < first[p["book_id"]]["page"]:
            first[p["book_id"]] = p
    out = []
    for book_id in sorted(first):
        meta, _, _ = ck.parse_page(first[book_id]["text"])
        out.append((book_id, meta.get("doc_title", "")))
    return out


def catalog_text(catalog: list[tuple[str, str]]) -> str:
    return "\n".join(f"{book_id} | {title}" for book_id, title in catalog)


def build_messages(spec: dict, catalog: str, question: str) -> list[tuple[str, str]]:
    # `{catalog}`만 바꾼다(문안 안의 JSON 중괄호 때문에 str.format을 쓰지 않는다)
    return [("system", spec["system"].replace("{catalog}", catalog)), ("human", spec["user_template"].replace("{question}", question))]


def parse_doc_ids(text: str | None, valid_ids: set[str], limit: int = MAX_DOCS) -> tuple[list[str], int, str | None]:
    """(고른 문서 번호, 목록에 없어 버린 번호 수, 실패 사유). 중복은 한 번만, 최대 limit개."""
    try:
        raw = json.loads(text or "")["doc_ids"]
        if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
            return [], 0, "doc_ids가 문자열 목록이 아님"
    except (ValueError, KeyError, TypeError) as exc:
        return [], 0, f"출력을 해석하지 못함({type(exc).__name__})"
    ids, invalid = [], 0
    for x in raw:
        x = re.sub(r"\s+", "", x).upper()
        if x not in valid_ids:
            invalid += 1
        elif x not in ids:
            ids.append(x)
    return ids[:limit], invalid, None


def build_llm(*, timeout: float = 60, max_completion_tokens: int = 2048, reasoning_effort: str = "low", seed: int = 20261111):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=os.environ["OPENAI_MODEL"], max_retries=2, timeout=timeout, reasoning_effort=reasoning_effort,
                      max_completion_tokens=max_completion_tokens, seed=seed, model_kwargs={"response_format": RESPONSE_FORMAT})


def route(llm, spec: dict, catalog: str, valid_ids: set[str], question: str) -> dict:
    """질문 1개의 문서를 고른다. 예외는 던지지 않고 `error`에 담는다."""
    from src.generation.generator import _text, _usage

    started = time.perf_counter()
    row = {"question": question, "doc_ids": [], "invalid_ids": 0, "raw": None, "usage": None, "error": None}
    try:
        msg = llm.invoke(build_messages(spec, catalog, question))
        row["raw"], row["usage"] = _text(msg.content), _usage(msg)
        row["doc_ids"], row["invalid_ids"], row["error"] = parse_doc_ids(row["raw"], valid_ids)
    except Exception as exc:  # noqa: BLE001 - 호출 실패가 평가 전체를 멈추지 않게 기록한다
        row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    row["seconds"] = round(time.perf_counter() - started, 2)
    return row
