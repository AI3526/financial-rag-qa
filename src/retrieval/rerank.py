"""Reranker와 Hybrid + Reranking (STEP 6, 08 §10, 06 §6·§11.1).

Hybrid의 RRF 상위 20개 후보를 질문과 함께 Cross-Encoder(`BAAI/bge-reranker-v2-m3`)에 넣어 관련성 점수를 다시 계산하고,
점수가 높은 순서로 최종 k=5를 고른다. 설정 조정(후보 수, 다른 모델)은 [확장]이며 공식 비교 설정과 분리한다.

- Reranker 입력 텍스트는 임베딩·BM25와 같은 검색용 텍스트(`retrieval_text`: 머리말·제목·소제목 + 본문)다.
- 점수가 같으면 재정렬 전 순위(rank_before)가 앞선 쪽이 먼저다(결과가 항상 같다).
- Reranker 오류는 검색 오류(`RetrievalError`, API 500 `retrieval_error`)다. Hybrid 순서로 몰래 대체하지 않는다(방식 비교가
  섞이지 않도록).
- 기록: `reranked = [{chunk_id, rank_before, rank_after, score}]` (09 §9). 점수는 모델이 돌려준 값 그대로이며 다른 방식의 점수와
  단순 비교하지 않는다.
- 평가 정답·근거 원문은 이 모듈에 들어오지 않는다. 입력은 질문과 후보 청크뿐이다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from src.ingest import chunk as ck
from src.retrieval.dense import InvalidQuery, RetrievalError, RetrievalResult, result_output
from src.retrieval.trace import traced_retrieval

RERANKER_ID = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True)
class RerankConfig:
    candidates: int = 20  # Hybrid 결합 결과 중 재정렬할 후보 수(06 초기값)
    final_k: int = 5


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder 래퍼. 모델은 생성할 때 한 번만 읽는다."""

    def __init__(self, model_id: str = RERANKER_ID, device: str | None = None, batch_size: int = 16):
        import torch
        from sentence_transformers import CrossEncoder

        from src.index.embedding import cached_revision

        self.model_id = model_id
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CrossEncoder(model_id, device=self.device)
        self.revision = cached_revision(model_id)

    def score(self, question: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self.model.predict([(question, t) for t in texts], batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]


def rerank_chunks(question: str, chunks: list[dict], scorer: Callable[[str, list[str]], list[float]]) -> list[dict]:
    """후보 청크를 점수 순으로 재정렬한다. 반환: rank_after 순서의 청크(rank_before, rank_after, rerank_score 포함)."""
    if not chunks:
        return []
    texts = [ck.retrieval_text({"chunk_text": c["text"], "metadata": c.get("metadata") or {}}) for c in chunks]
    scores = scorer(question, texts)
    if len(scores) != len(chunks):
        raise RetrievalError(f"Reranker 점수 수 불일치: 후보 {len(chunks)}개, 점수 {len(scores)}개")
    order = sorted(range(len(chunks)), key=lambda i: (-scores[i], chunks[i]["rank"], chunks[i]["chunk_id"]))
    return [{**chunks[i], "rank_before": chunks[i]["rank"], "rank": new_rank, "rank_after": new_rank,
             "score": scores[i], "rerank_score": scores[i]} for new_rank, i in enumerate(order, start=1)]


class RerankRetriever:
    """Hybrid 후보를 Reranker로 재정렬해 최종 k를 돌려준다(`rag_method=hybrid_rerank`)."""

    def __init__(self, hybrid, reranker, config: RerankConfig | None = None, *, langfuse=None):
        self.hybrid, self.reranker = hybrid, reranker
        self.config = config or RerankConfig()
        self.langfuse = langfuse

    def describe(self) -> dict:
        d = dict(self.hybrid.describe()) if hasattr(self.hybrid, "describe") else {}
        d.update({"reranker": getattr(self.reranker, "model_id", None), "reranker_revision": getattr(self.reranker, "revision", None),
                  "rerank_candidates": self.config.candidates, "final_k": self.config.final_k})
        return d

    def retrieve(self, question: str, k: int | None = None) -> RetrievalResult:
        c = self.config
        return traced_retrieval(
            self.langfuse, "rerank-retrieve", {"question": question, "k": c.final_k if k is None else k},
            {"method": "hybrid_rerank", "reranker": getattr(self.reranker, "model_id", None),
             "reranker_revision": getattr(self.reranker, "revision", None), "rerank_candidates": c.candidates},
            lambda: self._retrieve(question, k), result_output)

    def _retrieve(self, question: str, k: int | None) -> RetrievalResult:
        c = self.config
        k = c.final_k if k is None else k
        if not isinstance(question, str) or not question.strip():
            raise InvalidQuery("질문이 비어 있습니다.")
        if k < 1:
            raise InvalidQuery("k는 1 이상이어야 합니다.")
        t0 = time.perf_counter()
        base = self.hybrid.retrieve(question, k=c.candidates)  # RRF 결합 상위 candidates개(Dense·BM25 오류는 그대로 전달)
        t1 = time.perf_counter()
        try:
            ranked = rerank_chunks(question.strip(), base.chunks, self.reranker.score)
        except RetrievalError:
            raise
        except Exception as exc:  # noqa: BLE001 - 원인 종류와 무관하게 retrieval_error로 통일한다
            raise RetrievalError(f"Reranker {type(exc).__name__}: {str(exc)[:200]}") from exc
        t2 = time.perf_counter()
        cands = dict(base.candidates or {})
        cands["reranked"] = [{"chunk_id": r["chunk_id"], "rank_before": r["rank_before"], "rank_after": r["rank_after"],
                              "score": r["score"]} for r in ranked]
        return RetrievalResult(
            question=question, method="hybrid_rerank", k=k, chunks=ranked[:k], candidates=cands,
            candidate_k=base.candidate_k,
            durations_ms={**{n: v for n, v in base.durations_ms.items() if n != "total"},
                          "hybrid_total": base.durations_ms.get("total"), "rerank": round((t2 - t1) * 1000, 1),
                          "total": round((t2 - t0) * 1000, 1)})
