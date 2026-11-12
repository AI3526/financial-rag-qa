"""분해 계획(query-plan-v1) 기반 자리 배정 검색 — 확장 브랜치 어려운 경로 (EXTENSION_INTERMEDIATE_LOG).

- 원래 질문은 Hybrid + 재정렬(후보 `pool`)로 검색하고, 분해 질의(저장된 결과)마다 **재정렬 없이 Hybrid**로 검색한다(명확한 짧은 질의라 재정렬이 필요 없고 시간이 줄어든다).
- 분기 표기 병기(`period_notation.add_quarter_notation`)는 **키워드(BM25) 검색에만** 적용한다: 질의 전체에 붙이면 임베딩이 흐려져 "N분기"로 적힌 본문 쪽(IF17 p9 등)이 1위에서 밀려났다(plan-v1 첫 실행 진단). Dense는 원래 질의, BM25는 병기한 질의로 검색해 RRF로 합친다.
- 최종 k개: 분해 질의마다 상위 `sub_slots`개(전체 `sub_cap`개 이하, 중복은 한 번만)를 먼저 확보하고 나머지를 원래 질문의 재정렬 상위로 채운다(`slot_query.allocate_slots`).
- 분해 질의가 없는 질문은 원래 질문의 재정렬 상위 k개로, 기존 Hybrid+Rerank와 같다.
"""
from __future__ import annotations

import time

from src.retrieval.dense import InvalidQuery, RetrievalResult
from src.retrieval.hybrid import rrf_fuse
from src.retrieval.period_notation import add_quarter_notation
from src.retrieval.slot_query import allocate_slots


class PlanSlotRetriever:
    def __init__(self, rerank, hybrid, lookup: dict[str, list[str]], *, pool: int = 40, sub_slots: int = 2, sub_cap: int = 6, quarter_notation: bool = True):
        self.rerank, self.hybrid, self.lookup = rerank, hybrid, lookup
        self.pool, self.sub_slots, self.sub_cap, self.quarter_notation = pool, sub_slots, sub_cap, quarter_notation
        self.config = rerank.config
        self.langfuse = None

    def describe(self) -> dict:
        d = dict(self.rerank.describe())
        d.update({"plan_slot": True, "plan_pool": self.pool, "plan_sub_slots": self.sub_slots, "plan_sub_cap": self.sub_cap, "plan_quarter_notation": self.quarter_notation,
                  "questions_with_plan_queries": sum(1 for v in self.lookup.values() if v)})
        return d

    def _hybrid(self, query: str, top: int) -> list[dict]:
        """Dense는 원래 질의, BM25는 분기 표기를 병기한 질의로 검색해 RRF로 합친다(표기 병기가 없으면 일반 Hybrid와 같다)."""
        h = self.hybrid
        keyword_query = add_quarter_notation(query) if self.quarter_notation else query
        if keyword_query == query:
            return h.retrieve(query, k=top).chunks
        c = h.config
        dense = h.dense.retrieve(query, k=c.dense_candidates).chunks
        sparse = h.sparse.retrieve(keyword_query, k=c.sparse_candidates).chunks
        return rrf_fuse(dense, sparse, c.rrf_k, c.fused_top)[:top]

    def retrieve(self, question: str, k: int | None = None) -> RetrievalResult:
        if not isinstance(question, str) or not question.strip():
            raise InvalidQuery("질문이 비어 있습니다.")
        k = self.config.final_k if k is None else k
        t0 = time.perf_counter()
        plan = [q for q in self.lookup.get(question, []) if q != question]
        original = self.rerank.retrieve(question, k=self.pool)
        queries = list(plan)
        sub_lists = [self._hybrid(q, self.sub_slots + 6) for q in queries]  # 중복으로 건너뛰는 몫까지 여유 있게 받는다
        chunks = allocate_slots(original.chunks, sub_lists, k, self.sub_slots, self.sub_cap)
        return RetrievalResult(question=question, method="hybrid_rerank", k=k, chunks=chunks, candidate_k=original.candidate_k,
                               candidates={"queries": [question] + queries, "keyword_queries": [add_quarter_notation(q) if self.quarter_notation else q for q in queries], "slot_source": [c["slot_source"] for c in chunks]},
                               durations_ms={"total": round((time.perf_counter() - t0) * 1000, 1)})
