"""중대 오류 분류 입력·검증·표본 선정·대표 사례 선정 (STEP 12, D-045). 모델·네트워크 없이 확인한다."""
from src.evaluation import critical as c


def _judge(**scores):
    base = {k: {"status": "scored", "score": 1, "votes": [1, 1, 1], "reason": "r"} for k in ("facts", "conditions", "evidence", "decision")}
    for k, v in scores.items():
        base[k] = {"status": "scored" if v is not None else "unevaluable", "score": v, "votes": [], "reason": "r"}
    return {"items": base}


def test_has_zero_ignores_unevaluable_and_not_applicable():
    assert c.has_zero(_judge(facts=0)) and not c.has_zero(_judge())
    assert not c.has_zero(_judge(facts=None)) and not c.has_zero(None)


def _rec(rid, status="answered"):
    return {"request_id": rid, "question": "q", "response": {"status": status, "answer": "a", "citations": []}, "final_context": {"chunks": []}}


def test_classification_cases_keeps_zero_answers_and_counts_skips():
    runs = {("Q1", "dense"): _rec("r1"), ("Q2", "dense"): _rec("r2"), ("Q3", "dense"): _rec("r3", "error")}
    refs = {q: {"expected_action": "answer", "required_facts": ["f"], "forbidden_claims": []} for q in ("Q1", "Q2", "Q3")}
    judge = {"r1": _judge(facts=0), "r2": _judge(), "r3": _judge(facts=0)}
    cases, skipped = c.classification_cases(runs, refs, judge, {"Q1": "development"})
    assert [x["request_id"] for x in cases] == ["r1"] and skipped == {"service_error": 1, "no_zero": 1}
    assert cases[0]["required_facts"] == ["f"] and "judge" in cases[0]


def _row(rid, method, possible=(), cause="none"):
    return {"request_id": rid, "rag_method": method, "question_id": rid, "cause": cause, "reason": "x", "classified_by": "claude-code-session",
            "criteria": {k: {"possible": k in possible, "evidence": "e"} for k in c.CRITERIA}}


def test_validate_classification_flags_missing_duplicate_extra_and_bad_format():
    ok = [_row("a", "dense"), _row("b", "dense")]
    assert c.validate_classification(ok, ["a", "b"]) == []
    probs = c.validate_classification([_row("a", "dense"), _row("a", "dense"), {**_row("z", "dense"), "cause": "bogus"}], ["a", "b"])
    assert any("두 번" in p for p in probs) and any("누락: b" in p for p in probs) and any("대상이 아닌" in p for p in probs) and any("cause" in p for p in probs)
    bad = _row("a", "dense")
    bad["criteria"]["C1"] = {"possible": "yes", "evidence": 1}
    assert c.validate_classification([bad], ["a"])


def test_candidate_is_recomputed_from_criteria_not_from_a_candidate_flag():
    r = {**_row("a", "dense", possible=("C2",)), "candidate": False}
    assert c.is_candidate(r) and not c.is_candidate(_row("b", "dense"))


def test_sample_size_is_max_5_or_20_percent_capped_at_pool():
    assert [c.sample_size(n) for n in (0, 3, 5, 15, 25, 40)] == [0, 3, 5, 5, 5, 8]


def test_select_review_takes_all_candidates_and_a_reproducible_sample():
    rows = [_row(f"d{i:02d}", "dense") for i in range(12)] + [_row("dc", "dense", ("C1",))] + [_row(f"h{i}", "hybrid") for i in range(3)]
    a, b = c.select_review(rows), c.select_review(list(reversed(rows)))
    assert a == b and a["candidates"] == ["dc"]
    assert len(a["sample"]["dense"]) == 5 and set(a["sample"]["dense"]) <= {f"d{i:02d}" for i in range(12)}
    assert len(a["sample"]["hybrid"]) == 3 and a["sample"]["hybrid_rerank"] == []
    assert a["counts"]["dense"] == {"candidates": 1, "candidates_reviewed": 1, "candidates_cap_applied": False, "not_candidates": 12, "sampled": 5}


def test_representative_cases_pick_largest_changes_with_id_tiebreak_and_skip_missing():
    def row(q, m, **kw):
        base = {"question_id": q, "rag_method": m, "status_match": 1, "faith": None, "j_facts": 1, "j_conditions": 1, "j_evidence": 1, "j_decision": 1}
        return (q, m), {**base, **kw}
    table = dict([row("Q1", "dense", j_facts=0, j_decision=0), row("Q1", "hybrid"), row("Q2", "dense", j_facts=0), row("Q2", "hybrid"),
                  row("Q3", "dense"), row("Q3", "hybrid", j_facts=0), row("Q4", "dense"), row("Q4", "hybrid"), row("Q5", "dense", j_facts=None)])
    reps = [r for r in c.representative_cases(table, per_direction=2) if r["pair"] == "dense->hybrid"]
    assert [(r["direction"], r["question_id"], r["score_change"]) for r in reps] == [("improved", "Q1", 2), ("improved", "Q2", 1), ("worsened", "Q3", -1)]


def test_select_review_caps_candidates_per_method_with_the_same_count_and_reports_totals():
    rows = [_row(f"d{i:02d}", "dense", ("C1",)) for i in range(14)] + [_row(f"h{i:02d}", "hybrid", ("C2",)) for i in range(10)] + [_row("hr", "hybrid_rerank", ("C3",))]
    a = c.select_review(rows)
    assert c.select_review(list(reversed(rows))) == a
    by = lambda p: [i for i in a["candidates"] if i.startswith(p)]  # noqa: E731
    assert len(by("d")) == 10 and len(by("h0")) + len(by("h1")) == 10 and by("hr") == ["hr"]
    assert len(a["candidates_not_reviewed"]["dense"]) == 4 and a["candidates_not_reviewed"]["hybrid"] == []
    assert a["counts"]["dense"]["candidates"] == 14 and a["counts"]["dense"]["candidates_reviewed"] == 10 and a["counts"]["dense"]["candidates_cap_applied"]
    assert not a["counts"]["hybrid"]["candidates_cap_applied"]
