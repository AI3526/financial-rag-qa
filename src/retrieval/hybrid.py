"""Hybrid Retrieval: Dense + BM25를 RRF로 결합 (STEP 5, 08 §9, 06 §5.4·§11.1).

초기값(06): Dense 후보 20 + BM25 후보 20 → RRF(k=60) → 결합 상위 20 → 최종 k=5. 검색 필터는 쓰지 않는다.
설정 조정(후보 수, RRF 상수 등)은 [확장]이며 공식 비교 설정과 분리한다.

RRF: 점수 체계가 다른 두 결과를 순위로만 결합한다. `score(d) = Σ 1 / (rrf_k + rank_i(d))`.
한쪽 결과에만 있는 청크도 포함된다. 동점은 두 결과에서의 더 좋은(작은) 순위, 그다음 chunk_id로 정해 결과가 항상 같다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from src.retrieval.dense import InvalidQuery, RetrievalResult, result_output
from src.retrieval.trace import traced_retrieval


@dataclass(frozen=True)
class HybridConfig:
    dense_candidates: int = 20
    sparse_candidates: int = 20
    rrf_k: int = 60
    fused_top: int = 20
    final_k: int = 5


def rrf_fuse(dense: list[dict], sparse: list[dict], rrf_k: int = 60, top: int = 20) -> list[dict]:
    """두 순위 목록을 RRF로 결합한다. 반환: rank 1부터, score(RRF), dense_rank·sparse_rank(없으면 None)."""
    rows: dict[str, dict] = {}
    for source, items in (("dense_rank", dense), ("sparse_rank", sparse)):
        for item in items:
            row = rows.setdefault(item["chunk_id"], {**{k: item[k] for k in ("chunk_id", "book_id", "file_name", "page", "text", "metadata")},
                                                     "dense_rank": None, "sparse_rank": None, "score": 0.0})
            row[source] = item["rank"]
            row["score"] += 1.0 / (rrf_k + item["rank"])
    ordered = sorted(rows.values(), key=lambda r: (-r["score"], min(x for x in (r["dense_rank"], r["sparse_rank"]) if x),
                                                   r["chunk_id"]))[:top]
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank
    return ordered


def _brief(chunks: list[dict]) -> list[dict]:
    return [{"chunk_id": c["chunk_id"], "book_id": c["book_id"], "page": c["page"], "rank": c["rank"],
             "score": c["score"]} for c in chunks]


class HybridRetriever:
    def __init__(self, dense, sparse, config: HybridConfig | None = None, *, langfuse=None):
        self.dense, self.sparse = dense, sparse
        self.config = config or HybridConfig()
        self.langfuse = langfuse

    def retrieve(self, question: str, k: int | None = None) -> RetrievalResult:
        c = self.config
        return traced_retrieval(
            self.langfuse, "hybrid-retrieve", {"question": question, "k": c.final_k if k is None else k},
            {"method": "hybrid", "fusion": "rrf", "rrf_k": c.rrf_k, "dense_candidates": c.dense_candidates,
             "sparse_candidates": c.sparse_candidates, "fused_top": c.fused_top,
             "tokenizer": getattr(self.sparse, "tokenizer_name", None)},
            lambda: self._retrieve(question, k), result_output)

    def describe(self) -> dict:
        """run 레코드에 남길 검색 설정(재현용)."""
        c = self.config
        return {"fusion": "rrf", "rrf_k": c.rrf_k, "dense_candidates": c.dense_candidates,
                "sparse_candidates": c.sparse_candidates, "fused_top": c.fused_top, "final_k": c.final_k,
                "bm25_tokenizer": getattr(self.sparse, "tokenizer_name", None)}

    def _retrieve(self, question: str, k: int | None) -> RetrievalResult:
        c = self.config
        k = c.final_k if k is None else k
        if not isinstance(question, str) or not question.strip():
            raise InvalidQuery("질문이 비어 있습니다.")
        if k < 1:
            raise InvalidQuery("k는 1 이상이어야 합니다.")
        t0 = time.perf_counter()
        dense = self.dense.retrieve(question, k=c.dense_candidates)  # 저장소·임베딩 오류는 RetrievalError로 그대로 전달된다
        t1 = time.perf_counter()
        sparse = self.sparse.retrieve(question, k=c.sparse_candidates)
        t2 = time.perf_counter()
        fused = rrf_fuse(dense.chunks, sparse.chunks, c.rrf_k, c.fused_top)
        t3 = time.perf_counter()
        final = [{**row, "rank": i} for i, row in enumerate(fused[:k], start=1)]
        return RetrievalResult(
            question=question, method="hybrid", k=k, chunks=final, candidate_k=c.dense_candidates,
            candidates={"dense": _brief(dense.chunks), "sparse": _brief(sparse.chunks), "fused": _brief(fused)},
            durations_ms={"dense": round((t1 - t0) * 1000, 1), "sparse": round((t2 - t1) * 1000, 1),
                          "fusion": round((t3 - t2) * 1000, 1), "total": round((t3 - t0) * 1000, 1)})
