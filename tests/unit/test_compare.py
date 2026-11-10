"""공식 비교 통계 순수 함수 (STEP 12, 12_ACCEPTANCE_CRITERIA §4.1·§4.2)."""
from src.evaluation import compare as c


def test_group_stats_counts_missing_instead_of_zero():
    g = c.group_stats({"a": 1, "b": 0, "c": None, "d": None})
    assert g == {"n": 2, "missing": 2, "mean": 0.5}
    assert c.group_stats({"a": None}) == {"n": 0, "missing": 1, "mean": None}


def test_paired_stats_counts_better_worse_same_and_excludes_unpaired():
    a = {"q1": 0, "q2": 1, "q3": 1, "q4": 0, "q5": None, "q6": 1}
    b = {"q1": 1, "q2": 1, "q3": 0, "q4": 1, "q5": 1, "q7": 1}   # q5는 a가 없음, q6는 b가 없음, q7은 a에 없음
    s = c.paired_stats(a, b)
    assert s["n"] == 4 and (s["better"], s["worse"], s["same"]) == (2, 1, 1)  # q1, q4 좋아짐 / q3 나빠짐 / q2 같음
    assert abs(s["mean_diff"] - 0.25) < 1e-12 and s["mean_a"] == 0.5 and s["mean_b"] == 0.75 and s["excluded"] == 2
    lo, hi = s["ci95"]
    assert lo <= s["mean_diff"] <= hi


def test_similar_threshold_is_plus_minus_005_inclusive():
    same = {f"q{i}": 1 for i in range(20)}
    one_off = {**same, "q0": 0}                          # 20문항 중 1개 차이 = 0.05
    assert c.paired_stats(same, one_off)["similar"] is True
    two_off = {**same, "q0": 0, "q1": 0}                  # 0.10
    assert c.paired_stats(same, two_off)["similar"] is False
    thirty = {f"q{i}": 1 for i in range(30)}
    assert c.paired_stats(thirty, {**thirty, "q0": 0, "q1": 0})["similar"] is False  # 0.0667 > 0.05(한 문제는 0.033)
    assert c.paired_stats(thirty, {**thirty, "q0": 0})["similar"] is True


def test_paired_stats_with_no_common_questions_is_all_none():
    s = c.paired_stats({"a": 1}, {"b": 1})
    assert s["n"] == 0 and s["mean_diff"] is None and s["ci95"] is None and s["similar"] is None


def test_bootstrap_ci_is_reproducible_and_brackets_the_mean():
    diffs = [1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 1.0, 0.0]
    assert c.bootstrap_ci(diffs) == c.bootstrap_ci(diffs) and c.bootstrap_ci(diffs, seed=1) != c.bootstrap_ci(diffs, seed=2)
    lo, hi = c.bootstrap_ci(diffs)
    assert lo < sum(diffs) / len(diffs) < hi and c.bootstrap_ci([]) is None
    assert c.bootstrap_ci([0.5] * 10) == (0.5, 0.5)


def run(status="answered", total=1000.0, calls=((100, 20),), cost=0.0003, cost_status="complete"):
    return {"response": {"status": status}, "performance": {"total_duration_ms": total, "retrieval_duration_ms": 100.0, "generation_duration_ms": 800.0},
            "llm_calls": [{"input_tokens": i, "output_tokens": o} for i, o in calls], "cost": cost, "cost_status": cost_status}


def test_operational_separates_errors_unknown_usage_and_unknown_cost():
    runs = [run(), run(total=3000.0, calls=((200, 40), (200, 40)), cost=0.0006),   # 재시도 1번
            run(status="error", total=500.0, calls=(), cost=None, cost_status="unknown"),   # 오류 요청은 호출·비용 기록이 없다
            run(calls=((None, None),), cost=None, cost_status="unknown")]
    op = c.operational(runs)
    assert (op["requests"], op["ok"], op["errors"]) == (4, 3, 1) and op["error_rate"] == 0.25
    assert abs(op["mean_total_ms"] - (1000 + 3000 + 1000) / 3) < 1e-9          # 오류 요청은 응답시간 평균에서 제외
    assert op["input_tokens"] == 100 + 400 and op["output_tokens"] == 20 + 80  # 사용량을 확인한 요청만 더함
    assert (op["usage_known_requests"], op["usage_unknown_requests"]) == (2, 2)
    assert abs(op["cost_usd"] - 0.0009) < 1e-12 and (op["cost_known_requests"], op["cost_unknown_requests"]) == (2, 2)
    assert op["retries"] == 1


def test_operational_with_no_requests_is_all_none_or_zero():
    op = c.operational([])
    assert op["requests"] == 0 and op["error_rate"] is None and op["mean_total_ms"] is None and op["input_tokens"] == 0


def test_post_scores_plan_skips_missing_values_and_uses_stable_ids():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("post_scores", Path(__file__).resolve().parents[2] / "scripts" / "post_scores.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = [{"request_id": "r1", "trace_id": "t1", "question_id": "Q1", "rag_method": "dense", "status_match": 1, "hit": None, "recall": None,
             "faith": 0.5, "j_facts": 0, "j_conditions": None, "j_evidence": 1, "j_decision": 1},
            {"request_id": "r2", "trace_id": None, "question_id": "Q2", "rag_method": "dense", "status_match": 1}]
    items = mod.plan(rows)
    assert {i["score_id"] for i in items} == {"r1:status_match", "r1:faith", "r1:j_facts", "r1:j_evidence", "r1:j_decision"}
    assert all(i["trace_id"] == "t1" for i in items) and {i["value"] for i in items if i["name"] == "j_facts"} == {0.0}
