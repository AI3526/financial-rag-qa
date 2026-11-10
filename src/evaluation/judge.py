"""LLM-as-a-Judge 4항목 채점 (STEP 11, 과제 04_step6, DECISIONS D-040).

항목: facts(사실), conditions(조건), evidence(근거), decision(판단). 항목마다 점수 1 / 0 / 해당 없음(`not_applicable`) /
평가 불가(`unevaluable`)와 짧은 이유를 남긴다.

- 모델은 1, 0, 해당 없음만 낸다. **평가 불가는 코드가 정한다**(채점 호출 실패, 출력 형식 오류, 서비스 오류로 답변이 없음).
  0이나 1로 바꾸지 않는다.
- 서비스 상태가 기대 동작과 다르면 decision은 0이다. 이 비교는 모델이 아니라 **코드가** 하고, 모델의 decision 출력은 덮어쓴다
  (모델 원래 출력은 `model_output`에 남긴다).
- 서비스 오류(`error`) 응답은 모델을 부르지 않는다: decision 0, 나머지 평가 불가(과제 97행).
- 참고 정답·필수 사실·금지 주장·기대 동작은 이 평가 단계에서만 입력한다. 서비스(검색·생성) 코드는 이 모듈을 쓰지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from src.generation.context import format_context

ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = ROOT / "prompts" / "judge_v1.json"
EVALUATION_DIR = ROOT / "data" / "finance_basic" / "evaluation"
REFERENCE_FILES = ("development.json", "additional.json", "expansion_validation.json")

ITEMS = ("facts", "conditions", "evidence", "decision")
ACTION_TO_STATUS = {"answer": "answered", "clarify": "needs_clarification", "abstain": "abstained"}  # 과제 103행
SERVICE_ERROR_REASON = "서비스 오류로 답변이 없어 확인할 수 없음"

_ITEM_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["score", "status", "reason"],
    "properties": {"score": {"type": ["integer", "null"], "enum": [1, 0, None]},
                   "status": {"type": "string", "enum": ["scored", "not_applicable"]},
                   "reason": {"type": "string"}}}
RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "judge_result", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": list(ITEMS), "properties": {k: _ITEM_SCHEMA for k in ITEMS}}}}


def load_judge_prompt(path: Path = PROMPT_PATH) -> dict:
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    return {"version": data["version"], "sha256": hashlib.sha256(raw).hexdigest(), "system": data["system"], "user": data["user"]}


def load_references(directory: Path = EVALUATION_DIR, files: tuple[str, ...] = REFERENCE_FILES) -> dict[str, dict]:
    """질문 ID -> 평가 항목(참고 정답 등). 파일이 겹치는 ID를 가지면 오류로 중단한다."""
    refs: dict[str, dict] = {}
    for name in files:
        for item in json.loads((directory / name).read_text(encoding="utf-8")):
            if item["id"] in refs:
                raise ValueError(f"평가 파일에서 질문 ID가 겹칩니다: {item['id']}")
            refs[item["id"]] = item
    return refs


def expected_status(expected_action: str) -> str:
    if expected_action not in ACTION_TO_STATUS:
        raise ValueError(f"알 수 없는 expected_action: {expected_action!r}")
    return ACTION_TO_STATUS[expected_action]


def status_match(expected_action: str, service_status: str) -> str:
    return "일치" if expected_status(expected_action) == service_status else "불일치"


def _lines(values: list[str] | None) -> str:
    return "\n".join(f"- {v}" for v in values) if values else "(없음)"


def format_citations(citations: list[dict] | None) -> str:
    if not citations:
        return "(없음)"
    return "\n".join(f"- {c['source_id']}: {c['file_name']} {c['page']}쪽" for c in citations)


def build_user_message(template: str, record: dict, ref: dict) -> str:
    """Judge 사용자 메시지. 최종 문맥은 서비스 모델이 실제로 받은 형식([S1] 본문) 그대로 넣는다."""
    resp = record["response"]
    return template.format(
        question=record["question"], status=resp.get("status"), answer=resp.get("answer") or "", reason=resp.get("reason") or "",
        citations=format_citations(resp.get("citations")),
        final_context=format_context((record.get("final_context") or {}).get("chunks", [])),
        expected_action=f"{ref['expected_action']} ({expected_status(ref['expected_action'])})",
        status_match=status_match(ref["expected_action"], resp.get("status")),
        reference_answer=ref.get("reference_answer") or "(없음)", required_facts=_lines(ref.get("required_facts")),
        forbidden_claims=_lines(ref.get("forbidden_claims")), grading_notes=ref.get("grading_notes") or "(없음)")


def _unevaluable(reason: str) -> dict:
    return {"status": "unevaluable", "score": None, "reason": reason, "decided_by": "code"}


def parse_judgement(parsed: Any) -> dict[str, dict]:
    """모델 출력(JSON)을 항목별 결과로 바꾼다. 형식이 맞지 않는 항목은 평가 불가다(0이나 1로 바꾸지 않는다)."""
    out: dict[str, dict] = {}
    for key in ITEMS:
        it = parsed.get(key) if isinstance(parsed, dict) else None
        score = it.get("score") if isinstance(it, dict) else None
        status = it.get("status") if isinstance(it, dict) else None
        reason = it.get("reason") if isinstance(it, dict) else None
        ok = (isinstance(reason, str) and reason.strip() != "" and
              ((status == "scored" and not isinstance(score, bool) and score in (0, 1)) or (status == "not_applicable" and score is None)))
        out[key] = ({"status": status, "score": score, "reason": reason.strip(), "decided_by": "model"} if ok
                    else _unevaluable(f"출력 형식 오류: {key} 항목을 해석하지 못함"))
    return out


def finalize(expected_action: str, service_status: str, judged: dict[str, dict] | None, failure: str | None = None) -> dict[str, dict]:
    """모델 결과(없을 수 있음)에 코드 규칙을 적용해 최종 4항목 결과를 만든다.

    - 서비스 오류: 모델 결과를 쓰지 않는다. decision 0, 나머지 평가 불가.
    - 상태 불일치: decision 0(코드). 모델 호출 실패 여부와 관계없이 결정된다.
    - 모델 결과가 없으면(호출 실패) 나머지는 평가 불가와 사유.
    """
    expected = expected_status(expected_action)
    if service_status == "error":
        items = {k: _unevaluable(SERVICE_ERROR_REASON) for k in ITEMS}
        items["decision"] = {"status": "scored", "score": 0, "decided_by": "code",
                             "reason": f"서비스 오류로 답변을 만들지 못함(기대 {expected_action}, 실제 error)"}
        return items
    items = judged if judged is not None else {k: _unevaluable(failure or "채점 호출 실패") for k in ITEMS}
    if service_status != expected:
        model_output = items["decision"] if judged is not None else None
        items = {**items, "decision": {"status": "scored", "score": 0, "decided_by": "code",
                                       "reason": f"상태 불일치(기대 {expected_action}→{expected}, 실제 {service_status})",
                                       **({"model_output": {k: model_output[k] for k in ("status", "score", "reason")}} if model_output else {})}}
    return items


def failed_reason(exc: BaseException) -> str:
    return f"채점 호출 실패: {type(exc).__name__}: {str(exc)[:160]}"


def build_judge_llm(model: str | None = None, *, timeout: float = 120, max_completion_tokens: int = 4096, reasoning_effort: str = "low"):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model or os.environ["OPENAI_MODEL"], max_retries=2, timeout=timeout, reasoning_effort=reasoning_effort,
                      max_completion_tokens=max_completion_tokens, model_kwargs={"response_format": RESPONSE_FORMAT})


def judge_one(llm, spec: dict, record: dict, ref: dict) -> dict:
    """요청 1건을 채점해 결과 행을 돌려준다. 예외는 평가 불가로 바꿔 담고 던지지 않는다."""
    from src.generation.generator import _text, _usage

    resp = record["response"]
    row: dict[str, Any] = {
        "request_id": record["request_id"], "trace_id": record.get("trace_id"), "question_id": record.get("question_id"),
        "rag_method": record["rag_method"], "evaluation_set_type": record.get("evaluation_set_type"),
        "expected_action": ref["expected_action"], "service_status": resp.get("status"),
        "status_match": status_match(ref["expected_action"], resp.get("status")), "prompt_version": spec["version"],
        "usage": None, "model_output_raw": None}
    judged, failure, started = None, None, time.perf_counter()
    if resp.get("status") != "error":
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            ai = llm.invoke([SystemMessage(content=spec["system"]), HumanMessage(content=build_user_message(spec["user"], record, ref))])
            row["usage"] = {**(_usage(ai) or {}), "calls": 1}
            if (getattr(ai, "response_metadata", None) or {}).get("finish_reason") == "length":
                failure = "채점 출력이 max_completion_tokens에서 잘림"
            else:
                row["model_output_raw"] = _text(ai.content)
                try:
                    judged = parse_judgement(json.loads(row["model_output_raw"]))
                except json.JSONDecodeError:
                    failure = "출력 형식 오류: JSON이 아님"
        except Exception as exc:  # noqa: BLE001 - 호출 실패는 평가 불가로 기록한다
            failure = failed_reason(exc)
    row["seconds"] = round(time.perf_counter() - started, 2)
    row["items"] = finalize(ref["expected_action"], resp.get("status"), judged, failure)
    return row


def aggregate(rows: list[dict]) -> dict:
    """방식별 항목 충족률. 충족률 = 1인 답변 수 ÷ 점수가 있는 답변 수. 해당 없음과 평가 불가는 평균에서 빼고 개수로 센다."""
    out: dict[str, Any] = {}
    for method in sorted({r["rag_method"] for r in rows}):
        mine = [r for r in rows if r["rag_method"] == method]
        per: dict[str, Any] = {}
        for key in ITEMS:
            sts = [r["items"][key] for r in mine]
            scored = [s for s in sts if s["status"] == "scored"]
            ones = sum(1 for s in scored if s["score"] == 1)
            per[key] = {"scored": len(scored), "ones": ones, "rate": (ones / len(scored)) if scored else None,
                        "not_applicable": sum(1 for s in sts if s["status"] == "not_applicable"),
                        "unevaluable": sum(1 for s in sts if s["status"] == "unevaluable")}
        out[method] = {"answers": len(mine), "service_errors": sum(1 for r in mine if r["service_status"] == "error"),
                       "status_mismatch": sum(1 for r in mine if r["status_match"] == "불일치"), "items": per}
    return out
