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


SPEC = j.load_judge_prompt("judge_v1")  # v1 동작이 그대로인지 확인하는 회귀 테스트용
SPEC2 = j.load_judge_prompt("judge_v2")


def test_prompt_file_hash_version_and_approved_pieces():
    spec = j.load_judge_prompt("judge_v1")
    assert spec["version"] == "judge-v1" and spec["diagnostic"] is None
    assert spec["sha256"] == hashlib.sha256((j.PROMPTS_DIR / "judge_v1.json").read_bytes()).hexdigest()
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


# ---------------------------------------------------------------- judge-v2 (D-044)
CHUNKS2 = [{"source_id": "S1", "file_name": "FB01.pdf", "page": 3, "text": "본문1"},
           {"source_id": "S2", "file_name": "FB04.pdf", "page": 14, "text": "본문2"}]


def diag(states):
    return {j.DIAGNOSTIC_KEY: [{"index": i, "in_context": s, "reason": f"{i}번 이유"} for i, s in enumerate(states, 1)]}


def good2(states=("yes", "no"), **over):
    return {**good(**over), **diag(states)}


def test_v2_prompt_file_has_approved_sections_and_declares_the_diagnostic():
    spec = SPEC2
    assert spec["version"] == "judge-v2" and spec["diagnostic"] == j.DIAGNOSTIC_KEY and j.DEFAULT_PROMPT == "judge_v2"
    assert spec["sha256"] == hashlib.sha256((j.PROMPTS_DIR / "judge_v2.json").read_bytes()).hexdigest()
    for piece in ("[근거 항목의 부정 진술 규칙]", "[필수 사실의 문맥 확인 (진단)]", "사용자 메시지의 '출처 표'", "이 진단은 네 항목의 점수에 영향을 주지 않습니다.",
                  '"공개하지 않았다", "확정할 수 없다"처럼 그 내용을 직접 설명하고 있어서', "확인 요청만 있거나 부정 진술만 있는 답변 등",
                  "그 내용이 최종 문맥에 들어 있는데도 없다고 했다면 evidence는 0입니다.", "evidence는\n  not_applicable입니다"):
        assert piece in spec["system"], piece
    assert "{source_table}" in spec["user"] and "필수 사실(번호):\n{required_facts}" in spec["user"]
    assert "evidence_groups" not in spec["system"] + spec["user"]  # 사용자 결정: 입력하지 않는다


def test_v2_differs_from_v1_only_in_the_approved_places():
    v1, v2 = SPEC["system"].split("\n"), SPEC2["system"].split("\n")
    removed = [l for l in v1 if l not in v2]
    # v1에서 바뀌어 사라진 줄: 첫 단락 두 줄, 근거 항목 첫 줄, evidence 해당 없음 줄. 나머지 문장은 글자 그대로 유지된다
    assert len(removed) == 4 and all(any(k in l for k in ("질문은 포함되지 않습니다.", "당신은 질문", "3. evidence", "evidence: 답변에")) for l in removed), removed
    assert all(l in v1 for l in v2 if l.startswith(("1. facts", "2. conditions", "4. decision")))
    user1, user2 = SPEC["user"].split("\n"), SPEC2["user"].split("\n")
    assert [l for l in user1 if l not in user2] == ["필수 사실: {required_facts}"]
    assert "[출처 표] (최종 문맥의 S번호별 문서와 쪽)" in user2


def test_response_format_follows_the_prompt_version():
    assert j.response_format_for(SPEC) is j.RESPONSE_FORMAT and j.response_format_for(SPEC2) is j.RESPONSE_FORMAT_V2
    sch = j.RESPONSE_FORMAT_V2["json_schema"]
    assert sch["strict"] is True and j.DIAGNOSTIC_KEY in sch["schema"]["required"]
    assert j.DIAGNOSTIC_KEY not in j.RESPONSE_FORMAT["json_schema"]["schema"]["properties"]


def test_v2_user_message_has_source_table_for_every_chunk_and_numbered_required_facts():
    rec = record(citations=[{"source_id": "S1", "book_id": "FB01", "file_name": "FB01.pdf", "page": 3}], chunks=CHUNKS2)
    msg = j.build_user_message(SPEC2["user"], rec, REF)
    assert "[출처 표] (최종 문맥의 S번호별 문서와 쪽)\n- S1: FB01.pdf 3쪽\n- S2: FB04.pdf 14쪽" in msg  # 인용하지 않은 S2도 표에 있다
    assert "필수 사실(번호):\n1. 사실 A\n2. 사실 B" in msg
    assert "[S1] 본문1\n\n[S2] 본문2" in msg  # 최종 문맥 본문은 모델이 본 그대로
    assert "정답근거원문" not in msg and "evidence_groups" not in msg
    # v1은 이전과 똑같이 글머리 목록이고 출처 표가 없다
    old = j.build_user_message(SPEC["user"], rec, REF)
    assert "- 사실 A\n- 사실 B" in old and "[출처 표]" not in old


def test_parse_context_check_accepts_valid_and_marks_bad_output_unevaluable():
    facts = ["A", "B"]
    ok = j.parse_context_check(good2(("yes", "no")), facts)
    assert ok["status"] == "scored" and ok["coverage"] == "some" and [f["fact"] for f in ok["facts"]] == ["A", "B"]
    assert j.parse_context_check(good2(("yes", "yes")), facts)["coverage"] == "all"
    assert j.parse_context_check(good2(("no", "no")), facts)["coverage"] == "none"
    assert j.parse_context_check(good2(("partial", "yes")), facts)["coverage"] == "some"
    bad = [{}, {j.DIAGNOSTIC_KEY: "x"}, diag(("yes",)), diag(("yes", "yes", "yes")),
           {j.DIAGNOSTIC_KEY: [{"index": 1, "in_context": "yes", "reason": "r"}, {"index": 3, "in_context": "no", "reason": "r"}]},
           {j.DIAGNOSTIC_KEY: [{"index": 1, "in_context": "yes", "reason": "r"}, {"index": 1, "in_context": "no", "reason": "r"}]},
           {j.DIAGNOSTIC_KEY: [{"index": 1, "in_context": "maybe", "reason": "r"}, {"index": 2, "in_context": "no", "reason": "r"}]},
           {j.DIAGNOSTIC_KEY: [{"index": True, "in_context": "yes", "reason": "r"}, {"index": 2, "in_context": "no", "reason": "r"}]},
           {j.DIAGNOSTIC_KEY: [{"index": 1, "in_context": "yes", "reason": " "}, {"index": 2, "in_context": "no", "reason": "r"}]}]
    for b in bad:
        r = j.parse_context_check(b, facts)
        assert r["status"] == "unevaluable" and r["facts"] == [], b  # 추측으로 채우지 않는다
    assert j.parse_context_check(diag(()), [])["status"] == "not_applicable"


def test_judge_one_v2_records_context_check_and_it_does_not_change_the_four_scores():
    llm = FakeLLM(good2(("yes", "no")))
    row = j.judge_one(llm, SPEC2, record(chunks=CHUNKS2), REF)
    assert [row["items"][k]["score"] for k in j.ITEMS] == [1, 1, 1, 1]
    assert row["prompt_version"] == "judge-v2" and row["context_check"]["status"] == "scored"
    assert row["context_check"]["facts"][1]["in_context"] == "no" and row["context_check"]["facts"][1]["fact"] == "사실 B"
    sent = "\n".join(m.content for m in llm.calls[0])
    assert "[출처 표]" in sent and "- S2: FB04.pdf 14쪽" in sent and "정답근거원문" not in sent


def test_judge_one_v2_bad_diagnostic_is_unevaluable_but_scores_stay():
    row = j.judge_one(FakeLLM(good()), SPEC2, record(chunks=CHUNKS2), REF)  # 진단 항목이 없는 출력
    assert row["items"]["facts"]["score"] == 1 and row["context_check"]["status"] == "unevaluable"


def test_judge_one_v2_failures_and_service_error_leave_context_check_unevaluable():
    row = j.judge_one(FakeLLM(exc=TimeoutError("boom")), SPEC2, record(chunks=CHUNKS2), REF)
    assert row["context_check"]["status"] == "unevaluable" and "채점 호출 실패" in row["context_check"]["reason"]
    llm = FakeLLM(good2())
    err = j.judge_one(llm, SPEC2, record(status="error", answer="", chunks=CHUNKS2), REF)
    assert llm.calls == [] and err["context_check"]["status"] == "unevaluable" and "서비스 오류" in err["context_check"]["reason"]


def test_judge_one_v1_has_no_context_check():
    row = j.judge_one(FakeLLM(good()), SPEC, record(), REF)
    assert row["context_check"] is None and row["prompt_version"] == "judge-v1"


def test_aggregate_context_splits_zero_fact_answers_by_whether_the_context_had_the_facts():
    def row(facts_score, states):
        cc = (j.parse_context_check(diag(states), ["A", "B"]) if states else j._context_check_unevaluable("x"))
        return {"rag_method": "dense", "service_status": "answered", "status_match": "일치", "context_check": cc,
                "items": j.parse_judgement(good(facts={"score": facts_score, "status": "scored", "reason": "r"}))}

    rows = [row(0, ("yes", "yes")),   # 필수 사실이 문맥에 모두 있었는데 놓침 → 모델 누락 유력
            row(0, ("no", "no")),     # 문맥에 없음 → 검색 실패 유력
            row(0, ("yes", "no")),    # 일부만 있음
            row(1, ("yes", "yes")),
            row(1, None)]             # 진단 평가 불가
    c = j.aggregate(rows)["dense"]["context_check"]
    assert c["scored"] == 4 and c["unevaluable"] == 1 and c["facts_total"] == 8
    assert (c["facts_yes"], c["facts_no"], c["facts_partial"]) == (5, 3, 0) and c["fact_yes_rate"] == 5 / 8
    assert (c["answers_all"], c["answers_some"], c["answers_none"]) == (2, 1, 1)
    assert c["facts_zero"] == {"answers": 3, "context_all": 1, "context_some": 1, "context_none": 1}
    # v1 행(진단 없음)은 None
    v1 = [{"rag_method": "dense", "service_status": "answered", "status_match": "일치", "context_check": None, "items": j.parse_judgement(good())}]
    assert j.aggregate(v1)["dense"]["context_check"] is None
