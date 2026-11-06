"""검색 지표 Hit@k, Recall@k (02_EVALUATION_PLAN.md §5).

- 기준은 `book_id`(문서) 단위다: `evidence_groups`의 모든 book_id를 합쳐 중복을 제거한 집합.
- 같은 문서의 여러 청크는 한 번만 센다. 참고 근거가 비어 있으면 N/A(None).
- 참고용으로 (book_id, page) 단위 지표도 같은 방식으로 계산한다(공식 지표는 문서 단위).
"""
from __future__ import annotations


def reference_books(evidence_groups: list[list[dict]]) -> set[str]:
    return {e["book_id"] for group in evidence_groups for e in group}


def reference_pages(evidence_groups: list[list[dict]]) -> set[tuple[str, int]]:
    return {(e["book_id"], e["page"]) for group in evidence_groups for e in group}


def hit_recall(retrieved: set, reference: set) -> tuple[int | None, float | None]:
    """(Hit, Recall). 참고 집합이 비어 있으면 (None, None)."""
    if not reference:
        return None, None
    found = retrieved & reference
    return (1 if found else 0), len(found) / len(reference)


def question_metrics(retrieved_chunks: list[dict], evidence_groups: list[list[dict]]) -> dict:
    """검색된 청크(book_id, page 포함)로 문서 단위(공식)와 페이지 단위(참고) 지표를 계산한다."""
    books = {c["book_id"] for c in retrieved_chunks}
    pages = {(c["book_id"], c["page"]) for c in retrieved_chunks}
    hit, recall = hit_recall(books, reference_books(evidence_groups))
    p_hit, p_recall = hit_recall(pages, reference_pages(evidence_groups))
    return {"hit": hit, "recall": recall, "page_hit": p_hit, "page_recall": p_recall}


def mean_ignoring_none(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
