"""서비스 실행 기록에 적용하는 공식 Hit@k·Recall@k (STEP 11)."""
import pytest

from src.evaluation import retrieval_metrics as rm

EV = [[{"book_id": "FB01", "page": 1, "quote": "q"}], [{"book_id": "FB02", "page": 3, "quote": "q"}]]  # 문서 A, B


def rec(chunks, dropped=(), retrieval=True, final_k=5, extra_candidates=()):
    cand = [{"chunk_id": f"{c['book_id']}-p{c['page']}-c1", "book_id": c["book_id"], "page": c["page"], "rank": i, "score": 1.0}
            for i, c in enumerate(list(chunks) + list(extra_candidates), start=1)]
    return {"retrieval": {"dense": cand, "sparse": [], "fused": [], "final_k": final_k} if retrieval else None,
            "final_context": {"chunks": [{"chunk_id": f"{c['book_id']}-p{c['page']}-c1", "book_id": c["book_id"], "page": c["page"], "text": "t"} for c in chunks],
                              "dropped_chunk_ids": list(dropped)} if retrieval else None}


def ch(book, page):
    return {"book_id": book, "page": page}


def test_spec_example_a_found_b_missing_gives_hit1_recall_half_and_duplicates_count_once():
    m = rm.evaluate_record(rec([ch("FB01", 1), ch("FB01", 5), ch("FB09", 2)]), EV)
    assert m["status"] == "scored" and m["hit"] == 1 and m["recall"] == 0.5 and m["k"] == 5
    assert m["selected_books"] == ["FB01", "FB09"]


def test_same_document_wrong_page_counts_for_document_metric_but_not_page_metric():
    m = rm.evaluate_record(rec([ch("FB01", 9), ch("FB02", 9)]), EV)
    assert m["hit"] == 1 and m["recall"] == 1.0 and m["page_hit"] == 0 and m["page_recall"] == 0.0


def test_empty_result_with_reference_is_zero_not_excluded():
    m = rm.evaluate_record(rec([]), EV)
    assert m["status"] == "scored" and m["hit"] == 0 and m["recall"] == 0.0


def test_empty_reference_is_not_applicable_with_reason_and_no_zero():
    m = rm.evaluate_record(rec([ch("FB01", 1)]), [], "clarify")
    assert m["status"] == "not_applicable" and m["hit"] is None and m["recall"] is None and "clarify" in m["reason"]


def test_missing_retrieval_record_is_unevaluable_not_zero():
    m = rm.evaluate_record(rec([], retrieval=False), EV)
    assert m["status"] == "unevaluable" and m["hit"] is None and m["recall"] is None


def test_dropped_chunks_count_as_selected_but_not_as_final_context():
    # 선택된 5개 중 FB02 쪽이 토큰 한도로 제외됨: 검색 Recall은 1.0, 최종 문맥 Recall은 0.5
    r = rec([ch("FB01", 1)], dropped=["FB02-p3-c1"], extra_candidates=[ch("FB02", 3)])
    m = rm.evaluate_record(r, EV)
    assert m["recall"] == 1.0 and m["final_context"]["recall"] == 0.5 and m["final_context"]["dropped"] == 1
    assert m["selected_chunks"] == 2


def test_dropped_chunk_location_falls_back_to_chunk_id_and_bad_id_raises():
    r = rec([ch("FB01", 1)], dropped=["FB02-p3-c7"])
    assert rm.selected_chunks(r)[1] == {"chunk_id": "FB02-p3-c7", "book_id": "FB02", "page": 3}
    with pytest.raises(ValueError):
        rm.selected_chunks(rec([ch("FB01", 1)], dropped=["weird"]))


def test_aggregate_excludes_na_and_unevaluable_and_keeps_k():
    def row(method, metrics):
        return {"rag_method": method, "metrics": metrics}

    a = rm.evaluate_record(rec([ch("FB01", 1)]), EV)          # hit 1, recall .5
    b = rm.evaluate_record(rec([ch("FB09", 1)]), EV)          # hit 0, recall 0
    na = rm.evaluate_record(rec([ch("FB01", 1)]), [], "abstain")
    bad = rm.evaluate_record(rec([], retrieval=False), EV)
    s = rm.aggregate_records([row("dense", x) for x in (a, b, na, bad)])["dense"]
    assert s["questions"] == 4 and s["scored"] == 2 and s["not_applicable"] == 1 and s["unevaluable"] == 1
    assert s["hit_at_k"] == 0.5 and s["recall_at_k"] == 0.25 and s["k"] == [5]
