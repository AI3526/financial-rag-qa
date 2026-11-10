"""Judge 4항목 채점 (STEP 11, D-040): 네트워크 없이 가짜 모델로 확인한다."""
import hashlib
import json

import pytest
from langchain_core.messages import AIMessage

from src.evaluation import judge as j

REF = {"id": "Q1", "question": "q", "expected_action": "answer", "reference_answer": "참고 답", "required_facts": ["사실 A", "사실 B"],
       "forbidden_claims": ["금지 X"], "grading_notes": "설명", "evidence_groups": [[{"book_id": "FB01", "page": 1, "quote": "정답근거원문"}]]}


def record(status="answered", answer="답변", citations=None, chunks=None, run_type="general_eval"):
    return {"request_id": "r1", "trace_id": "t1", "question_id": "Q1", "rag_method": "dense", "question": "질문?", "run_type": run_type,
            "evaluation_set_type": "development",
            "response": {"status": status, "answer": answer, "reason": "이유", "citations": citations or []},
            "final_context": {"chunks": chunks if chunks is not None else [{"source_id": "S1", "text": "본문1"}]}}


def good(**over):
    base = {k: {"score": 1, "status": "scored", "reason": f"{k} 이유 S1"} for k in j.ITEMS}
    base.update(over)
    return base


class FakeLLM:
    def __init__(self, content=None, exc=None, finish="stop"):
        self.content, self.exc, self.finish, self.calls = content, exc, finish, []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.exc:
            raise self.exc
        return AIMessage(content=self.content if isinstance(self.content, str) else json.dumps(self.content, ensure_ascii=False),
                         response_metadata={"finish_reason": self.finish},
                         usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                                         "output_token_details": {"reasoning": 5}, "input_token_details": {"cache_read": 10}})


SPEC = j.load_judge_prompt()


def test_prompt_file_hash_version_and_approved_pieces():
    spec = j.load_judge_prompt()
    assert spec["version"] == "judge-v1"
    assert spec["sha256"] == hashlib.sha256(j.PROMPT_PATH.read_bytes()).hexdigest()
    for ph in ("{question}", "{status}", "{answer}", "{reason}", "{citations}", "{final_context}", "{expected_action}", "{status_match}",
               "{reference_answer}", "{required_facts}", "{forbidden_claims}", "{grading_notes}"):
        assert ph in spec["user"]
    assert "evidence_groups" not in spec["system"] + spec["user"]  # 사용자 결정: 입력하지 않는다
    assert "상태가 기대 동작과 다르면 이유와 관계없이 0" in spec["system"]
    assert "질문은 포함되지 않습니다" in spec["system"]


def test_status_mapping_and_match():
    assert j.status_match("answer", "answered") == "일치"
    assert j.status_match("clarify", "needs_clarification") == "일치"
    assert j.status_match("abstain", "abstained") == "일치"
    assert j.status_match("answer", "abstained") == "불일치"
    assert j.status_match("clarify", "answered") == "불일치"
    assert j.status_match("answer", "error") == "불일치"
    with pytest.raises(ValueError):
        j.expected_status("weird")


def test_user_message_has_final_context_in_service_format_but_not_evidence_groups_or_prompt_internals():
    rec = record(citations=[{"source_id": "S1", "book_id": "FB01", "file_name": "FB01.pdf", "page": 3}],
                 chunks=[{"source_id": "S1", "text": "본문1"}, {"source_id": "S2", "text": "본문2"}])
    msg = j.build_user_message(SPEC["user"], rec, REF)
    assert "[S1] 본문1\n\n[S2] 본문2" in msg  # 서비스 모델이 받은 형식 그대로
    assert "- S1: FB01.pdf 3쪽" in msg
    assert "기대 동작: answer (answered)" in msg and "상태 비교: 일치" in msg
    assert "- 사실 A\n- 사실 B" in msg and "- 금지 X" in msg and "참고 답" in msg
    assert "정답근거원문" not in msg and "evidence_groups" not in msg


def test_user_message_for_empty_lists_and_empty_context():
    ref = {**REF, "required_facts": [], "forbidden_claims": None}
    msg = j.build_user_message(SPEC["user"], record(chunks=[]), ref)
    assert "필수 사실: (없음)" in msg and "금지 주장: (없음)" in msg and "인용: (없음)" in msg and "(검색된 자료 없음)" in msg


def test_parse_judgement_accepts_valid_and_marks_malformed_items_unevaluable_not_zero():
    parsed = good(conditions={"score": None, "status": "not_applicable", "reason": "조건 없음"},
                  evidence={"score": 2, "status": "scored", "reason": "x"},          # 범위 밖 점수
                  decision={"score": True, "status": "scored", "reason": "x"})      # bool은 점수가 아니다
    out = j.parse_judgement(parsed)
    assert out["facts"]["score"] == 1 and out["facts"]["status"] == "scored"
    assert out["conditions"] == {"status": "not_applicable", "score": None, "reason": "조건 없음", "decided_by": "model"}
    assert out["evidence"]["status"] == "unevaluable" and out["evidence"]["score"] is None
    assert out["decision"]["status"] == "unevaluable" and out["decision"]["score"] is None
    assert j.parse_judgement("not a dict")["facts"]["status"] == "unevaluable"
    assert j.parse_judgement(good(facts={"score": 1, "status": "not_applicable", "reason": "x"}))["facts"]["status"] == "unevaluable"
    assert j.parse_judgement(good(facts={"score": 1, "status": "scored", "reason": "  "}))["facts"]["status"] == "unevaluable"


def test_status_mismatch_is_decided_by_code_and_overrides_model_decision():
    judged = j.parse_judgement(good())  # 모델은 decision=1을 냈다
    items = j.finalize("abstain", "answered", judged)
    d = items["decision"]
    assert d["score"] == 0 and d["decided_by"] == "code" and "상태 불일치" in d["reason"]
    assert d["model_output"]["score"] == 1  # 원래 출력은 보존
    assert items["facts"]["score"] == 1 and items["evidence"]["score"] == 1  # 나머지 세 항목은 그대로 채점


def test_status_match_keeps_model_decision():
    items = j.finalize("answer", "answered", j.parse_judgement(good(decision={"score": 0, "status": "scored", "reason": "이유 부적절"})))
    assert items["decision"]["score"] == 0 and items["decision"]["decided_by"] == "model"


def test_service_error_decision_zero_rest_unevaluable_without_model_result():
    items = j.finalize("answer", "error", None)
    assert items["decision"]["score"] == 0 and items["decision"]["status"] == "scored" and items["decision"]["decided_by"] == "code"
    for k in ("facts", "conditions", "evidence"):
        assert items[k]["status"] == "unevaluable" and items[k]["score"] is None


def test_call_failure_is_unevaluable_for_all_items_when_status_matches():
    items = j.finalize("answer", "answered", None, "채점 호출 실패: X")
    assert all(items[k]["status"] == "unevaluable" and items[k]["score"] is None for k in j.ITEMS)


def test_call_failure_with_status_mismatch_still_gets_rule_based_decision_zero():
    items = j.finalize("answer", "abstained", None, "채점 호출 실패: X")
    assert items["decision"]["score"] == 0 and items["decision"]["decided_by"] == "code" and "model_output" not in items["decision"]
    assert all(items[k]["status"] == "unevaluable" for k in ("facts", "conditions", "evidence"))


def test_judge_one_success_records_usage_and_sends_reference_data_only_to_judge():
    llm = FakeLLM(good())
    row = j.judge_one(llm, SPEC, record(), REF)
    assert row["items"]["facts"]["score"] == 1 and row["status_match"] == "일치"
    assert row["usage"]["input_tokens"] == 100 and row["usage"]["reasoning_tokens"] == 5 and row["usage"]["calls"] == 1
    sent = "\n".join(m.content for m in llm.calls[0])
    assert "사실 A" in sent and "정답근거원문" not in sent and "evidence_groups" not in sent


def test_judge_one_failures_become_unevaluable_never_zero_or_one():
    for llm, expect in ((FakeLLM(exc=TimeoutError("boom")), "채점 호출 실패"), (FakeLLM(content="not json"), "JSON이 아님"),
                        (FakeLLM(content=good(), finish="length"), "잘림")):
        row = j.judge_one(llm, SPEC, record(), REF)
        assert all(row["items"][k]["status"] == "unevaluable" and row["items"][k]["score"] is None for k in j.ITEMS)
        assert expect in row["items"]["facts"]["reason"]


def test_judge_one_service_error_does_not_call_model():
    llm = FakeLLM(good())
    row = j.judge_one(llm, SPEC, record(status="error", answer=""), REF)
    assert llm.calls == [] and row["usage"] is None
    assert row["items"]["decision"]["score"] == 0 and row["items"]["facts"]["status"] == "unevaluable"


def test_aggregate_excludes_na_and_unevaluable_from_rate_and_counts_them():
    rows = []
    for decision in (1, 1, 0):
        rows.append({"rag_method": "dense", "service_status": "answered", "status_match": "일치",
                     "items": j.parse_judgement(good(decision={"score": decision, "status": "scored", "reason": "r"}))})
    rows.append({"rag_method": "dense", "service_status": "answered", "status_match": "일치",
                 "items": j.parse_judgement(good(facts={"score": None, "status": "not_applicable", "reason": "r"}))})
    rows.append({"rag_method": "dense", "service_status": "error", "status_match": "불일치", "items": j.finalize("answer", "error", None)})
    s = j.aggregate(rows)["dense"]
    assert s["answers"] == 5 and s["service_errors"] == 1 and s["status_mismatch"] == 1
    assert s["items"]["decision"] == {"scored": 5, "ones": 3, "rate": 0.6, "not_applicable": 0, "unevaluable": 0}
    assert s["items"]["facts"] == {"scored": 3, "ones": 3, "rate": 1.0, "not_applicable": 1, "unevaluable": 1}


def test_references_load_without_duplicates_and_reference_data_is_not_in_service_modules():
    refs = j.load_references()
    assert len(refs) == 72 and refs["F-DEV-01"]["expected_action"] == "answer"
    with pytest.raises(ValueError):
        j.load_references(files=("development.json", "development.json"))
    # 서비스 코드는 평가 모듈을 가져오지 않는다(정답이 서비스 입력에 유입되지 않는다)
    from pathlib import Path

    for p in list((j.ROOT / "src" / "api").glob("*.py")) + list((j.ROOT / "src" / "generation").glob("*.py")) + list((j.ROOT / "src" / "retrieval").glob("*.py")):
        assert "src.evaluation" not in Path(p).read_text(encoding="utf-8"), p
