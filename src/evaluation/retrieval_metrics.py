"""검색 지표 Hit@k, Recall@k (02_EVALUATION_PLAN.md §5).

- 기준은 `book_id`(문서) 단위다: `evidence_groups`의 모든 book_id를 합쳐 중복을 제거한 집합.
- 같은 문서의 여러 청크는 한 번만 센다. 참고 근거가 비어 있으면 N/A(None).
- 참고용으로 (book_id, page) 단위 지표도 같은 방식으로 계산한다(공식 지표는 문서 단위).
"""
from __future__ import annotations

import re


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


# ---- 서비스 실행 기록(run 레코드)에 적용하는 공식 형식 (STEP 11, 과제 04_step6 "검색 지표") ----
_CHUNK_ID = re.compile(r"^(?P<book>.+)-p(?P<page>\d+)-c\d+$")


def _locator(record: dict) -> dict[str, tuple[str, int]]:
    """chunk_id -> (book_id, page). 검색 후보 목록과 최종 문맥 청크에서 모은다."""
    loc: dict[str, tuple[str, int]] = {}
    for key in ("dense", "sparse", "fused"):
        for c in (record.get("retrieval") or {}).get(key) or []:
            loc[c["chunk_id"]] = (c["book_id"], c["page"])
    for c in (record.get("final_context") or {}).get("chunks", []):
        loc[c["chunk_id"]] = (c["book_id"], c["page"])
    return loc


def selected_chunks(record: dict) -> list[dict] | None:
    """답변 생성 직전에 선택된 상위 k개 청크(문맥 토큰 한도로 제외되기 전). 검색 기록이 없으면 None.

    최종 문맥의 청크와 한도 때문에 제외된 청크(`dropped_chunk_ids`)를 합친 것이 선택된 k개다.
    """
    if record.get("retrieval") is None or record.get("final_context") is None:
        return None
    loc = _locator(record)
    fc = record["final_context"]
    out = [{"chunk_id": c["chunk_id"], "book_id": c["book_id"], "page": c["page"]} for c in fc.get("chunks", [])]
    for cid in fc.get("dropped_chunk_ids") or []:
        if cid in loc:
            book, page = loc[cid]
        else:
            m = _CHUNK_ID.match(cid)
            if not m:
                raise ValueError(f"chunk_id에서 문서와 쪽을 알 수 없습니다: {cid}")
            book, page = m["book"], int(m["page"])
        out.append({"chunk_id": cid, "book_id": book, "page": page})
    return out


def evaluate_record(record: dict, evidence_groups: list[list[dict]], expected_action: str | None = None) -> dict:
    """요청 1건의 검색 지표. 같은 참고 근거 목록을 세 방식에 쓴다(검색 결과에 맞춰 바꾸지 않는다).

    - status `scored`: Hit@k(문서 단위, 공식)·Recall@k와 (문서, 쪽) 단위 참고 지표. 검색 결과가 비어도 참고 문서가 있으면 0으로 계산한다.
    - status `not_applicable`: 참고 문서 목록이 비어 있음(이유를 적는다). 0점이 아니다.
    - status `unevaluable`: 서비스 오류 등으로 검색 기록이 없음. 0이나 1로 바꾸지 않는다.
    - `final_context`: 문맥 토큰 한도 이후 실제 최종 문맥에 남은 청크로 같은 지표를 따로 계산한다(필요한 근거가 문맥에 남았는지).
    """
    ref_books, ref_pages = reference_books(evidence_groups), reference_pages(evidence_groups)
    base = {"reference_books": sorted(ref_books), "reference_pages": sorted(ref_pages)}
    if not ref_books:
        return {**base, "status": "not_applicable", "hit": None, "recall": None,
                "reason": f"참고 근거 문서 목록이 비어 있음(기대 동작 {expected_action})" if expected_action else "참고 근거 문서 목록이 비어 있음"}
    selected = selected_chunks(record)
    if selected is None:
        return {**base, "status": "unevaluable", "hit": None, "recall": None, "reason": "검색 기록이 없음(서비스 오류 등)"}
    final_chunks = (record.get("final_context") or {}).get("chunks", [])
    m = question_metrics(selected, evidence_groups)
    f = question_metrics(final_chunks, evidence_groups)
    k = (record.get("retrieval") or {}).get("final_k") or len(selected)
    return {**base, "status": "scored", "reason": None, "k": k, "selected_chunks": len(selected),
            "selected_books": sorted({c["book_id"] for c in selected}), **m,
            "final_context": {"chunks": len(final_chunks), "dropped": len(selected) - len(final_chunks), **f}}


def aggregate_records(rows: list[dict]) -> dict:
    """방식별 평균. 점수가 있는 문항만 평균에 넣고 N/A와 평가 불가는 개수로 센다. k가 방식마다 다르면 그대로 드러낸다."""
    out: dict = {}
    for method in sorted({r["rag_method"] for r in rows}):
        mine = [r for r in rows if r["rag_method"] == method]
        scored = [r for r in mine if r["metrics"]["status"] == "scored"]
        fc = [r["metrics"]["final_context"] for r in scored]
        out[method] = {
            "questions": len(mine), "scored": len(scored),
            "not_applicable": sum(1 for r in mine if r["metrics"]["status"] == "not_applicable"),
            "unevaluable": sum(1 for r in mine if r["metrics"]["status"] == "unevaluable"),
            "k": sorted({r["metrics"]["k"] for r in scored}),
            "hit_at_k": mean_ignoring_none([r["metrics"]["hit"] for r in scored]),
            "recall_at_k": mean_ignoring_none([r["metrics"]["recall"] for r in scored]),
            "page_hit_at_k": mean_ignoring_none([r["metrics"]["page_hit"] for r in scored]),
            "page_recall_at_k": mean_ignoring_none([r["metrics"]["page_recall"] for r in scored]),
            "final_context_hit": mean_ignoring_none([x["hit"] for x in fc]),
            "final_context_recall": mean_ignoring_none([x["recall"] for x in fc]),
            "final_context_page_recall": mean_ignoring_none([x["page_recall"] for x in fc]),
            "questions_with_dropped_chunks": sum(1 for x in fc if x["dropped"] > 0),
        }
    return out
