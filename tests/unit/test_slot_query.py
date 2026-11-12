"""질의별 자리 배정 (확장 브랜치)."""
from src.retrieval.slot_query import allocate_slots


def lst(prefix, n):
    return [{"chunk_id": f"{prefix}{i}", "rank": i, "book_id": "B", "page": i, "text": "t"} for i in range(1, n + 1)]


def ids(chunks):
    return [c["chunk_id"] for c in chunks]


def test_without_sub_queries_equals_original_top_k():
    out = allocate_slots(lst("o", 20), [], 12)
    assert ids(out) == [f"o{i}" for i in range(1, 13)]
    assert all(c["slot_source"] == "original" for c in out)


def test_sub_queries_get_reserved_slots_and_original_fills_the_rest_in_order():
    out = allocate_slots(lst("o", 20), [lst("a", 10), lst("b", 10)], 12, sub_slots=3, sub_cap=6)
    assert len(out) == 12
    assert ids(out)[:6] == [f"o{i}" for i in range(1, 7)]  # 원래 질문 6개가 먼저
    assert ids(out)[6:] == ["a1", "a2", "a3", "b1", "b2", "b3"]  # 분해 질의마다 3개
    assert [c["rank"] for c in out] == list(range(1, 13))


def test_cap_splits_slots_evenly_when_many_sub_queries_and_duplicates_count_once():
    subs = [lst("a", 5), lst("b", 5), lst("c", 5)]
    out = allocate_slots(lst("o", 20), subs, 12, sub_slots=3, sub_cap=6)  # 3개 질의 → 질의당 2개
    assert ids(out)[6:] == ["a1", "a2", "b1", "b2", "c1", "c2"]
    shared = [{"chunk_id": "o1", "rank": 1, "book_id": "B", "page": 1, "text": "t"}] + lst("a", 5)
    out2 = allocate_slots(lst("o", 20), [shared], 12, sub_slots=3, sub_cap=6)
    assert len(ids(out2)) == len(set(ids(out2))) == 12  # 중복 청크는 한 번만
