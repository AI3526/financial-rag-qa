from src.evaluation import action_check as ac

QUESTION = {"id": "F-DEV-01", "question": "질문?", "reference_answer": "정답!", "expected_action": "answer",
            "evidence_groups": [[{"book_id": "FB01", "page": 1, "quote": "근거"}]], "required_facts": ["사실"]}


def test_request_body_contains_only_question_and_ids_never_evaluation_fields():
    body = ac.request_body(QUESTION)
    assert body == {"question": "질문?", "rag_method": "dense", "question_id": "F-DEV-01"}
    assert set(body) == {"question", "rag_method", "question_id"}  # 정답·근거·기대 행동 필드는 없다
    for leaked in ("정답!", "근거", "사실"):
        assert leaked not in str(body)


def test_expected_action_maps_to_api_status():
    assert ac.EXPECTED_STATUS == {"answer": "answered", "clarify": "needs_clarification", "abstain": "abstained"}


def test_compare_marks_match_and_mismatch_and_never_counts_error_as_match():
    ok = ac.compare("Q1", "clarify", {"status": "needs_clarification"}, 200)
    bad = ac.compare("Q2", "abstain", {"status": "answered"}, 200)
    err = ac.compare("Q3", "answer", {"status": "error"}, 502)
    assert ok["match"] and not bad["match"] and not err["match"] and err["http_status"] == 502


def test_summary_counts_by_action_and_returned_statuses():
    rows = [ac.compare("Q1", "answer", {"status": "answered"}, 200), ac.compare("Q2", "answer", {"status": "abstained"}, 200),
            ac.compare("Q3", "clarify", {"status": "needs_clarification"}, 200), ac.compare("Q4", "abstain", {"status": "error"}, 500)]
    s = ac.summarize(rows)
    assert (s["total"], s["match"], s["errors"]) == (4, 2, 1)
    assert s["by_action"]["answer"] == {"total": 2, "match": 1, "returned": {"abstained": 1, "answered": 1}}
    assert s["by_action"]["abstain"]["returned"] == {"error": 1}
