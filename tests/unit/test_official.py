"""공식 비교 실험의 동결 스냅샷, 사전 점검, 질문 목록, 결과 로더 (STEP 12, D-045). 네트워크·모델 없이 확인한다."""
import json

import pytest

from src.evaluation import official as o


def test_question_sets_are_60_questions_only_without_ground_truth():
    qs = o.load_question_sets()
    assert len(qs) == 60 and [q["evaluation_set_type"] for q in qs].count("development") == 30
    assert [q["evaluation_set_type"] for q in qs].count("additional") == 30
    assert all(set(q) == {"question_id", "question", "evaluation_set_type"} for q in qs)
    assert len({q["question_id"] for q in qs}) == 60


def test_question_sets_reject_a_questions_file_that_carries_extra_fields(tmp_path):
    for _, qf, af in o.SETS:
        (tmp_path / qf).write_text(json.dumps([{"id": "X1", "question": "q"}]), encoding="utf-8")
    (tmp_path / "development_questions.json").write_text(json.dumps([{"id": "X1", "question": "q", "reference_answer": "정답"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="질문 번호와 질문 외의 필드"):
        o.load_question_sets(tmp_path)


def _snap(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-6-luna")
    return o.snapshot(revision_of=lambda hf_id: f"rev-of-{hf_id}")


def test_snapshot_contains_everything_that_is_frozen(monkeypatch):
    s = _snap(monkeypatch)
    assert s["pipeline_settings"]["final_k"] == 5 and s["pipeline_settings"]["context_limit_tokens"] == 4000
    assert s["pipeline_settings"]["reasoning_effort"] == "low" and s["pipeline_settings"]["max_model_calls"] == 3
    assert s["answer_prompt"]["version"] == "answer-v2" and s["judge_prompt"]["version"] == "judge-v2"
    assert s["faithfulness_prompt"]["version"] == "faithfulness-statement-v1"
    assert all(len(s[k]["sha256"]) == 64 for k in ("answer_prompt", "judge_prompt", "faithfulness_prompt"))
    assert s["retrieval"]["hybrid"]["rrf_k"] == 60 and s["retrieval"]["rerank"]["candidates"] == 20 and s["retrieval"]["bm25"]["k1"] == 1.5
    assert s["retrieval"]["dense"]["collection"] == "basic__chunk-v1__e5large-v1"
    assert s["models"]["generation"] == "gpt-6-luna" and s["models"]["reranker"]["id"] == "BAAI/bge-reranker-v2-m3"
    assert s["evaluation"]["judge_passes"] == 3 and s["evaluation"]["faithfulness_passes"] == 3
    assert len(s["data"]["pdfs"]) == 15 and len(s["data"]["question_ids"]) == 60 and len(s["data"]["chunks_sha256"]) == 64
    assert s["packages"]["langfuse"] and s["python"].startswith("3.12")


def test_diff_snapshots_is_empty_for_equal_and_names_the_changed_path(monkeypatch):
    a = _snap(monkeypatch)
    b = json.loads(json.dumps(a))
    assert o.diff_snapshots(a, b) == []
    b["pipeline_settings"]["final_k"] = 8
    b["answer_prompt"]["sha256"] = "x" * 64
    del b["docs"]
    b["new"] = 1
    d = o.diff_snapshots(a, b)
    assert any(x.startswith("pipeline_settings.final_k: 5 → 8") for x in d)
    assert any(x.startswith("answer_prompt.sha256:") for x in d)
    assert any(x.startswith("docs: ") and "(현재 없음)" in x for x in d) and any(x.startswith("new: (동결에 없음)") for x in d)


def test_snapshot_survives_a_json_round_trip_without_false_differences(monkeypatch):
    """저장했다가 다시 읽은 동결 파일과 현재 스냅샷을 비교하므로, 튜플→리스트 같은 가짜 차이가 없어야 한다."""
    s = _snap(monkeypatch)
    assert isinstance(s["pipeline_settings"]["retry_delays_seconds"], list)
    assert o.diff_snapshots(json.loads(json.dumps(s, ensure_ascii=False)), _snap(monkeypatch)) == []


def test_parse_porcelain_ignores_blank_lines():
    assert o.parse_porcelain("") == [] and o.parse_porcelain(" M a.py\n?? b/\n\n") == [" M a.py", "?? b/"]


def test_check_runs_against_freeze_flags_version_drift_and_mixed_config():
    frozen = {"answer_prompt": {"sha256": "abc", "version": "answer-v2"},
              "retrieval": {"dense": {"embedding_version": "e5large-v1", "chunking_version": "chunk-v1", "dataset_version": "basic-v1"}},
              "models": {"generation": "gpt-6-luna"}}
    good = {"prompt_sha256": "abc", "prompt_version": "answer-v2", "embedding_version": "e5large-v1", "chunking_version": "chunk-v1",
            "dataset_version": "basic-v1", "model_id": "gpt-6-luna", "config_version": "cfg-1"}
    rec = lambda q, m, v: {"question_id": q, "rag_method": m, "versions": v}  # noqa: E731
    assert o.check_runs_against_freeze([rec("Q1", "dense", good), rec("Q2", "dense", good)], frozen) == []
    bad = o.check_runs_against_freeze([rec("Q1", "dense", good), rec("Q2", "dense", {**good, "prompt_sha256": "zzz", "config_version": "cfg-2"})], frozen)
    assert any("prompt_sha256" in p for p in bad) and any("config_version이 여러 개" in p for p in bad)


def test_verify_evaluation_manifest_checks_prompt_passes_limit_and_model():
    frozen = {"evaluation": dict(o.EVALUATION), "judge_prompt": {"sha256": "j"}, "faithfulness_prompt": {"sha256": "f"}, "models": {"generation": "m"}}
    ok_j = {"prompt": {"sha256": "j"}, "passes": 3, "max_output_tokens": 4096, "model": "m"}
    assert o.verify_evaluation_manifest("judge", ok_j, frozen) == []
    assert len(o.verify_evaluation_manifest("judge", {**ok_j, "passes": 1, "model": "x"}, frozen)) == 2
    ok_f = {"statement_prompt": {"sha256": "f"}, "passes": 3, "max_output_tokens": 4096, "tools": {"ragas": "0.4.3"}, "model": "m"}
    assert o.verify_evaluation_manifest("faithfulness", ok_f, frozen) == []
    assert any("ragas" in p for p in o.verify_evaluation_manifest("faithfulness", {**ok_f, "tools": {"ragas": "0.5"}}, frozen))
    assert o.verify_evaluation_manifest("other", ok_j, frozen)


def test_faith_per_request_means_scored_passes_and_keeps_na_and_unevaluable_apart():
    rows = [{"request_id": "a", "pass": 1, "status": "scored", "score": 1.0}, {"request_id": "a", "pass": 2, "status": "scored", "score": 0.5},
            {"request_id": "a", "pass": 3, "status": "unevaluable", "score": None},
            {"request_id": "b", "pass": 1, "status": "not_applicable", "score": None}, {"request_id": "b", "pass": 2, "status": "not_applicable", "score": None},
            {"request_id": "c", "pass": 1, "status": "unevaluable", "score": None}, {"request_id": "c", "pass": 2, "status": "not_applicable", "score": None}]
    f = o.faith_per_request(rows)
    assert f["a"]["status"] == "scored" and f["a"]["score"] == 0.75 and f["a"]["passes"] == [1.0, 0.5, "unevaluable"]
    assert f["b"] == {"status": "not_applicable", "score": None, "passes": ["not_applicable", "not_applicable"]}
    assert f["c"]["status"] == "unevaluable" and f["c"]["score"] is None


def test_build_table_uses_none_for_missing_values_and_never_zero():
    refs = {"Q1": {"expected_action": "answer"}, "Q2": {"expected_action": "clarify"}}
    runs = {("Q1", "dense"): {"request_id": "r1", "response": {"status": "answered"}},
            ("Q2", "dense"): {"request_id": "r2", "response": {"status": "answered"}}}
    retrieval = {("Q1", "dense"): {"status": "scored", "hit": 1, "recall": 0.5}, ("Q2", "dense"): {"status": "not_applicable", "hit": None, "recall": None}}
    faith = {"r1": {"status": "scored", "score": 0.9}, "r2": {"status": "not_applicable", "score": None}}
    sc = lambda v: {"status": "scored", "score": v}  # noqa: E731
    judge = {"r1": {"items": {"facts": sc(1), "conditions": {"status": "not_applicable", "score": None}, "evidence": sc(0), "decision": sc(1)}},
             "r2": {"items": {k: {"status": "unevaluable", "score": None} for k in o.JUDGE_ITEMS}}}
    t = o.build_table(runs, refs, retrieval, faith, judge, {"Q1": "development", "Q2": "additional"})
    r1, r2 = t[("Q1", "dense")], t[("Q2", "dense")]
    assert (r1["hit"], r1["recall"], r1["faith"], r1["j_facts"], r1["j_conditions"], r1["j_evidence"], r1["status_match"]) == (1, 0.5, 0.9, 1, None, 0, 1)
    assert r1["evaluation_set_type"] == "development" and r2["evaluation_set_type"] == "additional"
    assert r2["hit"] is None and r2["faith"] is None and r2["j_facts"] is None and r2["status_match"] == 0  # clarify 기대인데 answered
    assert r2["judge_status"]["facts"] == "unevaluable" and r2["faith_status"] == "not_applicable"


def test_load_runs_keeps_general_eval_only_and_rejects_duplicates(tmp_path):
    d = tmp_path / "exp"
    d.mkdir()
    rec = lambda q, m, rt="general_eval": {"question_id": q, "rag_method": m, "run_type": rt, "request_id": f"{q}{m}"}  # noqa: E731
    (d / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in (rec("Q1", "dense"), rec("Q1", "hybrid"), rec("Q9", "dense", "mock_error"))), encoding="utf-8")
    assert set(o.load_runs(["exp"], base=tmp_path)) == {("Q1", "dense"), ("Q1", "hybrid")}
    (d / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in (rec("Q1", "dense"), rec("Q1", "dense"))), encoding="utf-8")
    with pytest.raises(ValueError, match="두 번"):
        o.load_runs(["exp"], base=tmp_path)
