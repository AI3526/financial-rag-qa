import json

import pytest

from src.evaluation import faithfulness as fa


def v(statement, verdict, reason="이유"):
    return {"statement": statement, "reason": reason, "verdict": verdict}


# --- 프롬프트 파일 ---
def test_statement_prompt_file_has_version_hash_and_the_approved_additions():
    spec = fa.load_statement_prompt()
    assert spec["version"] == "faithfulness-statement-v1" and len(spec["sha256"]) == 64
    ins = spec["instruction"]
    # 라이브러리 원문 첫 문단이 그대로 있다
    assert ins.startswith("Given a question and an answer, analyze the complexity of each sentence in the answer.")
    assert "Ensure that no pronouns are used in any statement." in ins
    # 승인받은 추가 문단: 되묻기, 요청, 확인 불가 안내는 주장에서 제외, 없으면 빈 목록
    for clause in ("(a) ask the user for clarification", "(b) make a request or suggestion", "(c) say that some information is not available",
                   "return an empty list of statements"):
        assert clause in ins
    # 예시 2개: 라이브러리 원문(아인슈타인)과 추가 예시(빈 주장 목록)
    assert len(spec["examples"]) == 2
    assert spec["examples"][0]["question"] == "Who was Albert Einstein and what is he best known for?" and len(spec["examples"][0]["statements"]) == 4
    assert spec["examples"][1]["statements"] == [] and "알려 주시겠어요" in spec["examples"][1]["answer"]


def test_statement_prompt_contains_no_evaluation_answers():
    text = json.dumps(fa.load_statement_prompt(), ensure_ascii=False)
    for forbidden in ("reference_answer", "expected_action", "evidence_groups", "required_facts", "forbidden_claims"):
        assert forbidden not in text


# --- 결과 만들기 ---
def test_score_is_supported_over_total_and_keeps_every_reason():
    d = fa.build_detail(["a", "b", "c", "d"], [v("a", 1), v("b", 1), v("c", 1), v("d", 0, "문맥에 없음")])
    assert d["status"] == "scored" and d["score"] == 0.75 and (d["supported"], d["total"]) == (3, 4)
    assert [s["verdict"] for s in d["statements"]] == [1, 1, 1, 0] and d["statements"][3]["reason"] == "문맥에 없음"
    assert "뒷받침되지 않음: ['d']" in d["reason"] and "4개 중 3개" in d["reason"]


def test_all_supported_and_none_supported_are_real_scores_not_na():
    assert fa.build_detail(["a"], [v("a", 1)])["score"] == 1.0
    zero = fa.build_detail(["a"], [v("a", 0)])
    assert zero["status"] == "scored" and zero["score"] == 0.0  # 주장이 있고 모두 틀리면 진짜 0점


def test_no_statements_is_not_applicable_and_not_zero():
    d = fa.build_detail([], [])
    assert d["status"] == "not_applicable" and d["score"] is None and d["total"] == 0 and "사실 주장 없음" in d["reason"]


def test_statements_without_verdicts_are_unevaluable_not_a_score():
    d = fa.build_detail(["a", "b"], [])
    assert d["status"] == "unevaluable" and d["score"] is None


def test_verdict_values_are_normalized_to_0_or_1():
    d = fa.build_detail(["a", "b"], [v("a", True), v("b", None)])
    assert [s["verdict"] for s in d["statements"]] == [1, 0] and d["score"] == 0.5


def test_failed_call_is_unevaluable_without_a_score_and_without_leaking_long_messages():
    d = fa.failed_result(TimeoutError("x" * 500))
    assert d["status"] == "unevaluable" and d["score"] is None and "TimeoutError" in d["reason"] and len(d["reason"]) < 260


# --- 집계 ---
def row(rid, method, n, status, score=None):
    return {"request_id": rid, "rag_method": method, "pass": n, "status": status, "score": score}


def test_aggregate_excludes_na_and_unevaluable_from_the_mean_and_counts_them():
    rows = [row("r1", "dense", 1, "scored", 1.0), row("r2", "dense", 1, "scored", 0.5),
            row("r3", "dense", 1, "not_applicable"), row("r4", "dense", 1, "unevaluable")]
    s = fa.aggregate(rows)["dense"]
    assert s["answers"] == 4 and s["scored"] == 2 and s["not_applicable"] == 1 and s["unevaluable"] == 1
    assert s["mean"] == 0.75  # N/A와 평가 불가를 0으로 넣지 않는다


def test_aggregate_with_three_passes_reports_mean_range_and_disagreement():
    rows = [row("r1", "hybrid", n, "scored", sc) for n, sc in ((1, 0.75), (2, 1.0), (3, 0.75))]
    rows += [row("r2", "hybrid", n, "scored", 1.0) for n in (1, 2, 3)]
    s = fa.aggregate(rows)["hybrid"]
    assert s["answers"] == 2 and abs(s["mean"] - (((0.75 + 1.0 + 0.75) / 3) + 1.0) / 2) < 1e-9
    assert abs(s["max_range_across_passes"] - 0.25) < 1e-9 and s["answers_with_pass_disagreement"] == 1


def test_aggregate_mixed_passes_use_only_the_scored_passes():
    rows = [row("r1", "dense", 1, "scored", 0.5), row("r1", "dense", 2, "unevaluable"), row("r1", "dense", 3, "scored", 1.0)]
    s = fa.aggregate(rows)["dense"]
    assert s["scored"] == 1 and s["unevaluable"] == 0 and s["mean"] == 0.75  # 점수가 있는 회차의 평균


def test_aggregate_separates_methods_and_handles_all_na():
    rows = [row("a", "dense", 1, "not_applicable"), row("b", "hybrid", 1, "scored", 1.0)]
    out = fa.aggregate(rows)
    assert out["dense"]["mean"] is None and out["dense"]["not_applicable"] == 1 and out["hybrid"]["mean"] == 1.0
