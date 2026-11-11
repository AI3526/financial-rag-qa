"""문서 한정 검색을 더한 Hybrid — 확장 브랜치 방법 A (EXTENSION_INTERMEDIATE_LOG).

기존 Hybrid 후보(전체 코퍼스, 질문 하나)에 **라우팅이 고른 문서 안으로만 한정한 Hybrid 후보**를 더해 RRF로 합친다.
- 한정 검색은 같은 임베딩 행렬·BM25 색인에서 고른 문서의 청크만 남겨(나머지는 점수를 -inf로) Dense 상위·BM25 상위를 구하고 기존과 같은 RRF로 합친다.
- `HybridRetriever`와 같은 인터페이스라 `RerankRetriever(hybrid=…)`에 그대로 끼운다. 재정렬은 원래 질문으로 한다.
- 질문에 대응하는 문서는 `lookup`(질문 문자열 → 문서 번호 목록)에서 찾는다. 없거나 비어 있으면 기존 Hybrid와 같다(후보는 같은 `top`개).
- 후보 풀 크기는 `top`으로 고정한다. 효과가 후보 수가 아니라 문서 한정 후보에서 나오는지 보려고 단일 질의 기준선과 같은 풀 크기로 비교한다.
"""
from __future__ import annotations

import time

import numpy as np

from src.retrieval.dense import InvalidQuery, RetrievalResult
from src.retrieval.hybrid import _brief, rrf_fuse
from src.retrieval.multi_query import merge_rrf


class DocRoutedHybrid:
    def __init__(self, hybrid, lookup: dict[str, list[str]], *, top: int = 20, rrf_k: int = 60, restricted_top: int = 20):
        self.hybrid, self.lookup, self.top, self.rrf_k, self.restricted_top = hybrid, lookup, top, rrf_k, restricted_top
        self.config = hybrid.config
        self.langfuse = None
        self._masks: dict[frozenset, np.ndarray] = {}

    def describe(self) -> dict:
        d = dict(self.hybrid.describe())
        d.update({"doc_routed": True, "doc_routed_rrf_k": self.rrf_k, "doc_routed_pool": self.top, "doc_routed_restricted_top": self.restricted_top,
                  "questions_with_routed_docs": sum(1 for v in self.lookup.values() if v)})
        return d

    def _mask(self, docs: frozenset) -> np.ndarray:
        if docs not in self._masks:
            self._masks[docs] = np.array([c["book_id"] in docs for c in self.hybrid.sparse.chunks])
        return self._masks[docs]

    def _restricted(self, question: str, docs: list[str]) -> list[dict]:
        """고른 문서의 청크만으로 Dense·BM25 상위를 구해 RRF로 합친 순위 목록."""
        dense, sparse = self.hybrid.dense, self.hybrid.sparse
        mask = self._mask(frozenset(docs))
        n = self.restricted_top
        vec = np.asarray(dense.embedder.embed_queries([question.strip()])[0], dtype=np.float32)
        ds = np.where(mask, dense.index.vectors @ vec, -np.inf)
        bs = np.where(mask, np.asarray(sparse.bm25.get_scores(sparse.tokenize(question))), -np.inf)

        def as_chunks(scores: np.ndarray) -> list[dict]:
            order = [i for i in np.argsort(-scores, kind="stable")[:n] if np.isfinite(scores[i])]
            out = []
            for rank, i in enumerate(order, start=1):
                p = dense.index.payloads[i]
                out.append({"rank": rank, "score": float(scores[i]), "chunk_id": p["chunk_id"], "book_id": p["book_id"], "file_name": p["file_name"],
                            "page": p["page"], "text": p["chunk_text"], "metadata": p["metadata"]})
            return out

        return rrf_fuse(as_chunks(ds), as_chunks(bs), self.rrf_k, n)

    def retrieve(self, question: str, k: int | None = None) -> RetrievalResult:
        if not isinstance(question, str) or not question.strip():
            raise InvalidQuery("질문이 비어 있습니다.")
        k = self.config.final_k if k is None else k
        t0 = time.perf_counter()
        global_res = self.hybrid.retrieve(question, k=self.config.fused_top)
        docs = [d for d in self.lookup.get(question, [])]
        if docs:
            merged = merge_rrf([global_res.chunks, self._restricted(question, docs)], self.rrf_k, self.top)
        else:
            merged = [{**c, "rank": i} for i, c in enumerate(global_res.chunks[: self.top], start=1)]
        return RetrievalResult(
            question=question, method="hybrid", k=k, chunks=merged[:k], candidate_k=self.config.dense_candidates,
            candidates={"routed_docs": docs, "fused": _brief(merged)}, durations_ms={"total": round((time.perf_counter() - t0) * 1000, 1)})
