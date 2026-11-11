"""다중 질의 Hybrid 검색 — 확장 브랜치 5a (EXTENSION_INTERMEDIATE_LOG).

원래 질문과 분해한 검색 질의들을 각각 Hybrid로 검색하고, 질의별 순위 목록을 RRF로 합쳐 하나의 후보 목록을 만든다.
- `HybridRetriever`와 같은 인터페이스(`retrieve(question, k)`, `describe()`, `.candidates["fused"]`)라 `RerankRetriever(hybrid=…)`에 그대로 끼운다.
  재정렬은 질의가 아니라 **원래 질문**으로 한다(`RerankRetriever`가 받은 질문을 그대로 쓰기 때문).
- 질문에 대응하는 추가 질의는 `lookup`(질문 문자열 → 추가 질의 목록)에서 찾는다. 없으면 원래 질문만 검색한다(기존 Hybrid와 같음).
- 후보 풀 크기는 `top`으로 고정해(기본 20) 기존과 같게 두므로 효과는 후보 수가 아니라 질의 다양화에서 나온다.
"""
from __future__ import annotations

import time

from src.retrieval.dense import InvalidQuery, RetrievalResult
from src.retrieval.hybrid import _brief


def merge_rrf(lists: list[list[dict]], rrf_k: int = 60, top: int = 20) -> list[dict]:
    """여러 순위 목록을 RRF로 합친다. 점수는 목록마다 1/(rrf_k+순위)의 합이고 동점은 더 좋은 순위, chunk_id 순이다."""
    rows: dict[str, dict] = {}
    for items in lists:
        for item in items:
            row = rows.setdefault(item["chunk_id"], {**{k: item[k] for k in ("chunk_id", "book_id", "file_name", "page", "text", "metadata")},
                                                     "best_rank": item["rank"], "score": 0.0, "hits": 0})
            row["score"] += 1.0 / (rrf_k + item["rank"])
            row["best_rank"] = min(row["best_rank"], item["rank"])
            row["hits"] += 1
    ordered = sorted(rows.values(), key=lambda r: (-r["score"], r["best_rank"], r["chunk_id"]))[:top]
    return [{**r, "rank": i} for i, r in enumerate(ordered, start=1)]


class MultiQueryHybrid:
    def __init__(self, hybrid, lookup: dict[str, list[str]], *, top: int = 20, rrf_k: int = 60):
        self.hybrid, self.lookup, self.top, self.rrf_k = hybrid, lookup, top, rrf_k
        self.config = hybrid.config
        self.langfuse = None

    def describe(self) -> dict:
        d = dict(self.hybrid.describe())
        d.update({"multi_query": True, "multi_query_rrf_k": self.rrf_k, "multi_query_pool": self.top, "questions_with_extra_queries": sum(1 for v in self.lookup.values() if v)})
        return d

    def retrieve(self, question: str, k: int | None = None) -> RetrievalResult:
        if not isinstance(question, str) or not question.strip():
            raise InvalidQuery("질문이 비어 있습니다.")
        k = self.config.final_k if k is None else k
        t0 = time.perf_counter()
        queries = [question] + [q for q in self.lookup.get(question, []) if q != question]
        results = [self.hybrid.retrieve(q, k=self.config.fused_top) for q in queries]
        merged = merge_rrf([r.chunks for r in results], self.rrf_k, self.top)
        chunks = merged[:k]
        return RetrievalResult(
            question=question, method="hybrid", k=k, chunks=chunks, candidate_k=self.config.dense_candidates,
            candidates={"queries": queries, "fused": _brief(merged)}, durations_ms={"total": round((time.perf_counter() - t0) * 1000, 1)})


class MaxOverQueriesReranker:
    """확장 브랜치 (a): 다중 질의 후보 풀을 원래 질문만이 아니라 분해한 질의 각각으로도 재정렬해, 청크마다 가장 높은 점수를 쓴다.

    `RerankRetriever(reranker=…)`에 그대로 끼운다(점수 함수 `score(question, texts)`만 바꾸므로 핵심 코드는 그대로다).
    추가 질의가 없는 질문은 원래 점수와 완전히 같다. 같은 모델의 점수끼리 최댓값을 취한다.
    """

    def __init__(self, reranker, lookup: dict[str, list[str]]):
        self.reranker, self.lookup = reranker, lookup
        self.model_id = getattr(reranker, "model_id", None)
        self.revision = getattr(reranker, "revision", None)

    def score(self, question: str, texts: list[str]) -> list[float]:
        queries = [question] + [q for q in self.lookup.get(question, []) if q != question]
        per_query = [self.reranker.score(q, texts) for q in queries]
        return [max(s) for s in zip(*per_query)] if texts else []
