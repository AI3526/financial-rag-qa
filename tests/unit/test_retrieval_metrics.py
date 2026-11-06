from src.evaluation import retrieval_metrics as rm

GROUPS = [[{"book_id": "FB01", "page": 1, "quote": "q"}], [{"book_id": "FB02", "page": 3, "quote": "q"}],
          [{"book_id": "FB02", "page": 4, "quote": "q"}]]


def test_reference_sets_deduplicate_books_and_keep_pages():
    assert rm.reference_books(GROUPS) == {"FB01", "FB02"}
    assert rm.reference_pages(GROUPS) == {("FB01", 1), ("FB02", 3), ("FB02", 4)}


def test_hit_recall_is_document_level_and_counts_a_book_once():
    # 같은 문서의 청크 3개를 가져와도 FB02는 한 번만 센다
    chunks = [{"book_id": "FB02", "page": 3}, {"book_id": "FB02", "page": 9}, {"book_id": "FB02", "page": 4}]
    m = rm.question_metrics(chunks, GROUPS)
    assert m["hit"] == 1 and m["recall"] == 0.5  # 문서 2개 중 1개
    assert m["page_hit"] == 1 and m["page_recall"] == 2 / 3  # 페이지 3개 중 2개(참고용)


def test_miss_and_empty_retrieval_still_count_as_zero():
    m = rm.question_metrics([], GROUPS)
    assert m["hit"] == 0 and m["recall"] == 0.0


def test_no_reference_is_not_applicable():
    m = rm.question_metrics([{"book_id": "FB01", "page": 1}], [])
    assert m == {"hit": None, "recall": None, "page_hit": None, "page_recall": None}
    assert rm.mean_ignoring_none([1, None, 0]) == 0.5
    assert rm.mean_ignoring_none([None]) is None
