"""질의별 자리 배정 검색 — 확장 브랜치 (EXTENSION_INTERMEDIATE_LOG, 분해 질의의 검색 결과를 합치다 잃는 문제 시험).

배경: 질문 분해(5a)의 질의 하나만으로 검색하면 정답 쪽이 1위로 나오는 경우(IF-DEV07, IF-DEV02)에도, 질의들의 결과를 RRF로 한 순위표에 합친 뒤 상위만 남기면 그 쪽이 빠졌다.
여기서는 합치지 않고 **질의마다 검색+재정렬을 따로 하고 최종 자리를 나눠 준다**.
- 추가 질의가 없는 질문은 원래 질문 하나의 재정렬 결과 상위 k개로, 기존 Hybrid+Rerank와 같다.
- 추가 질의가 있으면 각 분해 질의의 재정렬 상위에서 질의당 `sub_slots`개(전체 `sub_cap`개 이하)를 먼저 확보하고, 나머지 자리는 원래 질문의 상위 청크로 채운다(중복은 한 번만).
- 최종 순서: 원래 질문 청크(순위 순) 다음에 분해 질의에서 확보한 청크. 총 k개(문맥 한도 4,000토큰은 파이프라인이 그대로 적용).
- `RerankRetriever`(질의별 재정렬을 수행)를 그대로 쓰며 평가 정답·근거는 쓰지 않는다.
"""
from __future__ import annotations

import time

from src.retrieval.dense import InvalidQuery, RetrievalResult


def allocate_slots(original: list[dict], sub_lists: list[list[dict]], k: int, sub_slots: int = 3, sub_cap: int = 6) -> list[dict]:
    """원래 질문 순위 목록과 분해 질의별 순위 목록에서 최종 k개를 고른다. 반환 청크에는 `slot_source`(original/sub{i})를 붙이고 rank를 1부터 다시 매긴다."""
    per_sub = max(1, min(sub_slots, sub_cap // len(sub_lists))) if sub_lists else 0
    seen: set[str] = set()
    sub_picks: list[dict] = []
    for i, items in enumerate(sub_lists, start=1):
        taken = 0
        for c in items:
            if taken >= per_sub:
                break
            if c["chunk_id"] in seen:
                continue
            seen.add(c["chunk_id"])
            sub_picks.append({**c, "slot_source": f"sub{i}"})
            taken += 1
    sub_picks = sub_picks[:k]
    orig_picks = []
    for c in original:
        if len(orig_picks) + len(sub_picks) >= k:
            break
        if c["chunk_id"] in seen:
            continue
        seen.add(c["chunk_id"])
        orig_picks.append({**c, "slot_source": "original"})
    return [{**c, "rank": i} for i, c in enumerate(orig_picks + sub_picks, start=1)]


class SlotQueryRerank:
    """`rerank`(RerankRetriever)를 질의마다 따로 실행하고 `allocate_slots`로 최종 청크를 정한다. `hybrid_rerank` 방식 자리에 그대로 끼운다."""

    def __init__(self, rerank, lookup: dict[str, list[str]], *, pool: int = 40, sub_slots: int = 3, sub_cap: int = 6):
        self.rerank, self.lookup, self.pool, self.sub_slots, self.sub_cap = rerank, lookup, pool, sub_slots, sub_cap
        self.config = rerank.config
        self.langfuse = None

    def describe(self) -> dict:
        d = dict(self.rerank.describe())
        d.update({"slot_query": True, "slot_pool": self.pool, "slot_sub_slots": self.sub_slots, "slot_sub_cap": self.sub_cap,
                  "questions_with_extra_queries": sum(1 for v in self.lookup.values() if v)})
        return d

    def retrieve(self, question: str, k: int | None = None) -> RetrievalResult:
        if not isinstance(question, str) or not question.strip():
            raise InvalidQuery("질문이 비어 있습니다.")
        k = self.config.final_k if k is None else k
        t0 = time.perf_counter()
        extras = [q for q in self.lookup.get(question, []) if q != question]
        original = self.rerank.retrieve(question, k=self.pool)
        sub_results = [self.rerank.retrieve(q, k=self.pool) for q in extras]
        chunks = allocate_slots(original.chunks, [r.chunks for r in sub_results], k, self.sub_slots, self.sub_cap)
        return RetrievalResult(question=question, method="hybrid_rerank", k=k, chunks=chunks, candidate_k=original.candidate_k,
                               candidates={"queries": [question] + extras, "slot_source": [c["slot_source"] for c in chunks]},
                               durations_ms={"total": round((time.perf_counter() - t0) * 1000, 1)})
