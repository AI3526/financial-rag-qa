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
PROMPTS_DIR = ROOT / "prompts"
DEFAULT_PROMPT = "judge_v2"  # 사용자가 승인한 최신 문안(D-044). judge_v1로 만든 결과는 그대로 보존한다
PROMPT_PATH = PROMPTS_DIR / f"{DEFAULT_PROMPT}.json"
DIAGNOSTIC_KEY = "required_facts_in_context"  # judge-v2의 진단 항목(점수에 영향 없음)
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
RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "judge_result", "strict": True, "schema": {  # judge-v1
    "type": "object", "additionalProperties": False, "required": list(ITEMS), "properties": {k: _ITEM_SCHEMA for k in ITEMS}}}}
_CONTEXT_CHECK_SCHEMA = {
    "type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["index", "in_context", "reason"],
        "properties": {"index": {"type": "integer"}, "in_context": {"type": "string", "enum": ["yes", "partial", "no"]},
                       "reason": {"type": "string"}}}}
RESPONSE_FORMAT_V2 = {"type": "json_schema", "json_schema": {"name": "judge_result_v2", "strict": True, "schema": {  # judge-v2
    "type": "object", "additionalProperties": False, "required": [*ITEMS, DIAGNOSTIC_KEY],
    "properties": {**{k: _ITEM_SCHEMA for k in ITEMS}, DIAGNOSTIC_KEY: _CONTEXT_CHECK_SCHEMA}}}}


def response_format_for(spec: dict) -> dict:
    """프롬프트 파일에 진단 항목이 선언되어 있으면 v2 출력 형식, 아니면 v1 출력 형식."""
    return RESPONSE_FORMAT_V2 if spec.get("diagnostic") == DIAGNOSTIC_KEY else RESPONSE_FORMAT


def load_judge_prompt(name_or_path: str | Path = DEFAULT_PROMPT) -> dict:
    """프롬프트 이름(`judge_v2`)이나 파일 경로로 읽는다. 파일의 sha256과 버전, 진단 항목 선언 여부를 함께 돌려준다."""
    path = name_or_path if isinstance(name_or_path, Path) else PROMPTS_DIR / f"{name_or_path}.json"
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    return {"version": data["version"], "sha256": hashlib.sha256(raw).hexdigest(), "system": data["system"], "user": data["user"],
            "diagnostic": data.get("diagnostic")}


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


def _numbered(values: list[str] | None) -> str:
    return "\n".join(f"{i}. {v}" for i, v in enumerate(values, start=1)) if values else "(없음)"


def format_citations(citations: list[dict] | None) -> str:
    if not citations:
        return "(없음)"
    return "\n".join(f"- {c['source_id']}: {c['file_name']} {c['page']}쪽" for c in citations)


def format_source_table(chunks: list[dict]) -> str:
    """최종 문맥의 **모든** 청크의 S번호별 문서와 쪽(judge-v2의 출처 표). Judge가 인용한 쪽을 검증할 수 있게 한다."""
    if not chunks:
        return "(없음)"
    return "\n".join(f"- {c['source_id']}: {c['file_name']} {c['page']}쪽" for c in chunks)


def build_user_message(template: str, record: dict, ref: dict) -> str:
    """Judge 사용자 메시지. 최종 문맥 본문은 서비스 모델이 실제로 받은 형식([S1] 본문) 그대로 넣는다.

    템플릿에 `{source_table}`이 있으면(judge-v2) 출처 표를 넣고 필수 사실에 번호를 붙인다(진단 항목이 번호로 답한다).
    없으면(judge-v1) 이전과 똑같이 `- 사실` 목록이다.
    """
    resp = record["response"]
    chunks = (record.get("final_context") or {}).get("chunks", [])
    v2 = "{source_table}" in template
    return template.format(
        question=record["question"], status=resp.get("status"), answer=resp.get("answer") or "", reason=resp.get("reason") or "",
        citations=format_citations(resp.get("citations")), final_context=format_context(chunks),
        source_table=format_source_table(chunks) if v2 else "",  # v1 입력에는 출처 표가 없다(쪽 정보가 없는 기록도 v1으로는 채점 가능)
        expected_action=f"{ref['expected_action']} ({expected_status(ref['expected_action'])})",
        status_match=status_match(ref["expected_action"], resp.get("status")),
        reference_answer=ref.get("reference_answer") or "(없음)",
        required_facts=(_numbered if v2 else _lines)(ref.get("required_facts")),
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


CONTEXT_STATES = ("yes", "partial", "no")


def _context_check_unevaluable(reason: str) -> dict:
    return {"status": "unevaluable", "reason": reason, "facts": [], "coverage": None}


def parse_context_check(parsed: Any, required_facts: list[str] | None) -> dict:
    """진단 항목(필수 사실이 최종 문맥에 있었는가)을 해석한다. **점수가 아니라 진단**이며 네 항목 점수에 영향을 주지 않는다.

    - 필수 사실이 없으면 해당 없음. 번호가 필수 사실과 정확히 일치하지 않거나 값이 형식에 맞지 않으면 평가 불가(추측으로 채우지 않는다).
    - coverage: 모두 yes면 `all`, 모두 no면 `none`, 그 밖은 `some`.
    """
    facts = list(required_facts or [])
    if not facts:
        return {"status": "not_applicable", "reason": "필수 사실이 없음", "facts": [], "coverage": None}
    raw = parsed.get(DIAGNOSTIC_KEY) if isinstance(parsed, dict) else None
    if not isinstance(raw, list):
        return _context_check_unevaluable("출력 형식 오류: 진단 항목이 없음")
    by_index: dict[int, dict] = {}
    for it in raw:
        idx = it.get("index") if isinstance(it, dict) else None
        state = it.get("in_context") if isinstance(it, dict) else None
        reason = it.get("reason") if isinstance(it, dict) else None
        if (isinstance(idx, bool) or not isinstance(idx, int) or state not in CONTEXT_STATES or not isinstance(reason, str)
                or not reason.strip() or idx in by_index):
            return _context_check_unevaluable("출력 형식 오류: 진단 항목을 해석하지 못함")
        by_index[idx] = {"index": idx, "in_context": state, "reason": reason.strip()}
    if set(by_index) != set(range(1, len(facts) + 1)):
        return _context_check_unevaluable("출력 형식 오류: 필수 사실 번호가 맞지 않음")
    out = [{**by_index[i], "fact": facts[i - 1]} for i in range(1, len(facts) + 1)]
    states = [f["in_context"] for f in out]
    coverage = "all" if all(s == "yes" for s in states) else ("none" if all(s == "no" for s in states) else "some")
    return {"status": "scored", "reason": None, "facts": out, "coverage": coverage}


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


def build_judge_llm(model: str | None = None, *, timeout: float = 120, max_completion_tokens: int = 4096, reasoning_effort: str = "low",
                    response_format: dict | None = None):
    """response_format은 프롬프트 버전에 맞는 출력 형식(`response_format_for(spec)`). 없으면 judge-v1 형식."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model or os.environ["OPENAI_MODEL"], max_retries=2, timeout=timeout, reasoning_effort=reasoning_effort,
                      max_completion_tokens=max_completion_tokens, model_kwargs={"response_format": response_format or RESPONSE_FORMAT})


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
    diag = spec.get("diagnostic") == DIAGNOSTIC_KEY
    judged, failure, started, context_check = None, None, time.perf_counter(), None
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
                    parsed_json = json.loads(row["model_output_raw"])
                    judged = parse_judgement(parsed_json)
                    if diag:
                        context_check = parse_context_check(parsed_json, ref.get("required_facts"))
                except json.JSONDecodeError:
                    failure = "출력 형식 오류: JSON이 아님"
        except Exception as exc:  # noqa: BLE001 - 호출 실패는 평가 불가로 기록한다
            failure = failed_reason(exc)
    row["seconds"] = round(time.perf_counter() - started, 2)
    row["items"] = finalize(ref["expected_action"], resp.get("status"), judged, failure)
    if diag:  # 진단은 점수가 아니므로 호출이 실패하거나 서비스 오류면 평가 불가로만 남긴다(네 항목 점수에는 영향 없음)
        row["context_check"] = context_check or _context_check_unevaluable(
            SERVICE_ERROR_REASON if resp.get("status") == "error" else (failure or "진단 항목을 얻지 못함"))
    else:
        row["context_check"] = None
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
                       "status_mismatch": sum(1 for r in mine if r["status_match"] == "불일치"), "items": per,
                       "context_check": _aggregate_context(mine)}
    return out


def _aggregate_context(mine: list[dict]) -> dict | None:
    """진단 항목 집계. judge-v1 행에는 진단이 없으므로 None. 사실=0인 답변을 문맥 포함 정도로 나눠 센다.

    - `facts_zero.context_all`: 사실=0인데 필수 사실이 **모두 문맥에 있었음**(모델이 놓친 쪽이 유력)
    - `facts_zero.context_some`/`context_none`: 일부/전부 **문맥에 없었음**(검색·문맥 구성 쪽이 유력). 유력일 뿐 확정은 아니다.
    """
    rows = [r for r in mine if r.get("context_check")]
    if not rows:
        return None
    cc = [r["context_check"] for r in rows]
    scored = [c for c in cc if c["status"] == "scored"]
    facts = [f for c in scored for f in c["facts"]]
    n = len(facts)
    counts = {s: sum(1 for f in facts if f["in_context"] == s) for s in CONTEXT_STATES}
    zero = [r for r in rows if r["context_check"]["status"] == "scored" and r["items"]["facts"]["status"] == "scored"
            and r["items"]["facts"]["score"] == 0]
    return {"scored": len(scored), "not_applicable": sum(1 for c in cc if c["status"] == "not_applicable"),
            "unevaluable": sum(1 for c in cc if c["status"] == "unevaluable"),
            "facts_total": n, "facts_yes": counts["yes"], "facts_partial": counts["partial"], "facts_no": counts["no"],
            "fact_yes_rate": (counts["yes"] / n) if n else None,
            "answers_all": sum(1 for c in scored if c["coverage"] == "all"), "answers_some": sum(1 for c in scored if c["coverage"] == "some"),
            "answers_none": sum(1 for c in scored if c["coverage"] == "none"),
            "facts_zero": {"answers": len(zero), "context_all": sum(1 for r in zero if r["context_check"]["coverage"] == "all"),
                           "context_some": sum(1 for r in zero if r["context_check"]["coverage"] == "some"),
                           "context_none": sum(1 for r in zero if r["context_check"]["coverage"] == "none")}}


# ---------------------------------------------------------------- 여러 회차 채점의 다수결 (D-044, D-045)
STATUSES = ("scored", "not_applicable", "unevaluable")


def majority_item(items: list[dict]) -> dict:
    """같은 답변·같은 항목의 회차별 결과를 다수결로 합친다. 0이나 1로 억지로 정하지 않는다.

    - 상태(점수 있음 / 해당 없음 / 평가 불가) 중 **절반을 넘는** 것이 그 상태가 된다. 없으면 평가 불가(회차가 갈림).
    - 점수 있음이 절반을 넘어도 **1과 0이 같은 수**(예: 2회만 점수가 있고 1과 0)이면 **동점이라 평가 불가**로 두고 이유가 `동점`으로 시작한다.
    - `votes`에 회차별 값(1, 0, "not_applicable", "unevaluable")을 남겨 회차 사이의 불일치를 볼 수 있게 한다.
    """
    n = len(items)
    votes = [i["score"] if i["status"] == "scored" else i["status"] for i in items]
    base = {"votes": votes, "decided_by": "majority"}
    top = next((s for s in STATUSES if sum(1 for i in items if i["status"] == s) * 2 > n), None)
    if top is None:
        return {**base, "status": "unevaluable", "score": None, "reason": f"회차 결과가 갈려 다수가 없음 {votes}"}
    if top != "scored":
        src = next(i for i in items if i["status"] == top)
        return {**base, "status": top, "score": None, "reason": src["reason"]}
    ones = sum(1 for i in items if i["status"] == "scored" and i["score"] == 1)
    zeros = sum(1 for i in items if i["status"] == "scored" and i["score"] == 0)
    if ones == zeros:
        return {**base, "status": "unevaluable", "score": None, "reason": f"동점: 점수가 있는 회차가 1과 0으로 갈림 {votes}"}
    score = 1 if ones > zeros else 0
    src = next(i for i in items if i["status"] == "scored" and i["score"] == score)
    return {**base, "status": "scored", "score": score, "reason": f"{src['reason']} [다수결 {max(ones, zeros)}/{n}회]"}


def majority_context(checks: list[dict | None]) -> dict | None:
    """회차별 필수 사실 진단을 합친다. 사실마다 yes/partial/no 다수결이고 동수면 `partial`이다.

    진단을 얻은 회차가 절반을 넘지 못하면 해당 없음(필수 사실 없음)이거나 평가 불가다.
    """
    cs = [c for c in checks if c]
    if not cs:
        return None
    n = len(cs)
    scored = [c for c in cs if c["status"] == "scored"]
    if len(scored) * 2 <= n:
        if sum(1 for c in cs if c["status"] == "not_applicable") * 2 > n:
            return next(c for c in cs if c["status"] == "not_applicable")
        return _context_check_unevaluable("회차 중 절반 이상에서 진단을 얻지 못함")
    facts = []
    for idx in range(len(scored[0]["facts"])):
        states = [c["facts"][idx]["in_context"] for c in scored]
        counts = {s: states.count(s) for s in CONTEXT_STATES}
        winners = [s for s in CONTEXT_STATES if counts[s] == max(counts.values())]
        state = winners[0] if len(winners) == 1 else "partial"
        src = next((c["facts"][idx] for c in scored if c["facts"][idx]["in_context"] == state), scored[0]["facts"][idx])
        facts.append({"index": src["index"], "fact": src["fact"], "in_context": state, "reason": src["reason"], "votes": states})
    sts = [f["in_context"] for f in facts]
    coverage = "all" if all(s == "yes" for s in sts) else ("none" if all(s == "no" for s in sts) else "some")
    return {"status": "scored", "reason": None, "facts": facts, "coverage": coverage}


_BASE_KEYS = ("request_id", "trace_id", "question_id", "rag_method", "evaluation_set_type", "expected_action", "service_status",
              "status_match", "prompt_version")


def combine_passes(rows: list[dict]) -> list[dict]:
    """(요청, 회차) 행들을 요청별 다수결 행으로 합친다. 각 행에는 `pass`(1부터)가 있어야 한다."""
    by_req: dict[str, list[dict]] = {}
    for r in rows:
        by_req.setdefault(r["request_id"], []).append(r)
    out = []
    for rs in by_req.values():
        rs = sorted(rs, key=lambda r: r["pass"])
        out.append({**{k: rs[0].get(k) for k in _BASE_KEYS}, "passes": len(rs),
                    "items": {k: majority_item([r["items"][k] for r in rs]) for k in ITEMS},
                    "context_check": majority_context([r.get("context_check") for r in rs])})
    return sorted(out, key=lambda r: (str(r["question_id"]), r["rag_method"]))


def aggregate_passes(pass_rows: list[dict], majority_rows: list[dict]) -> dict:
    """다수결 집계(선택·충족률에 쓰는 값)와 회차별 충족률, 회차 사이의 범위, 판정이 갈린 답변 수, 동점 수를 함께 낸다."""
    passes = sorted({r["pass"] for r in pass_rows})
    out: dict[str, Any] = {"passes": passes, "majority": aggregate(majority_rows),
                           "per_pass": {k: aggregate([r for r in pass_rows if r["pass"] == k]) for k in passes},
                           "range": {}, "disagreement": {}, "ties": {}}
    for m in sorted({r["rag_method"] for r in majority_rows}):
        mine = [r for r in majority_rows if r["rag_method"] == m]
        out["range"][m], out["disagreement"][m], out["ties"][m] = {}, {}, {}
        for key in ITEMS:
            rates = [out["per_pass"][k][m]["items"][key]["rate"] for k in passes if m in out["per_pass"][k]]
            rates = [x for x in rates if x is not None]
            out["range"][m][key] = (max(rates) - min(rates)) if rates else None
            out["disagreement"][m][key] = sum(1 for r in mine if len({str(v) for v in r["items"][key]["votes"]}) > 1)
            out["ties"][m][key] = sum(1 for r in mine if r["items"][key]["reason"].startswith("동점"))
    return out
