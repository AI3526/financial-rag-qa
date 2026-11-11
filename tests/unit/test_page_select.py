"""쪽 단위 선택 (확장 브랜치 4단계)."""
import pytest

from src.retrieval.page_select import select_top_pages


def _c(cid, book, page, rank):
    return {"chunk_id": cid, "book_id": book, "page": page, "rank": rank}


CHUNKS = [_c("a1", "B", 1, 1), _c("a2", "B", 1, 2), _c("b1", "B", 2, 3), _c("a3", "B", 1, 4), _c("c1", "C", 1, 5), _c("d1", "B", 3, 6)]


def test_cap_one_keeps_only_the_best_chunk_per_page_and_fills_with_next_pages():
    out = select_top_pages(CHUNKS, k=4, cap=1)
    assert [c["chunk_id"] for c in out] == ["a1", "b1", "c1", "d1"] and [c["rank"] for c in out] == [1, 2, 3, 4]


def test_cap_two_allows_a_second_chunk_from_the_same_page_but_not_a_third():
    assert [c["chunk_id"] for c in select_top_pages(CHUNKS, k=5, cap=2)] == ["a1", "a2", "b1", "c1", "d1"]


def test_large_cap_equals_the_original_top_k_in_order():
    assert [c["chunk_id"] for c in select_top_pages(CHUNKS, k=3, cap=5)] == ["a1", "a2", "b1"]


def test_same_page_number_in_different_books_is_not_a_duplicate_and_short_input_is_ok():
    out = select_top_pages([_c("x", "B", 1, 1), _c("y", "C", 1, 2)], k=5, cap=1)
    assert len(out) == 2
    with pytest.raises(ValueError):
        select_top_pages(CHUNKS, cap=0)
