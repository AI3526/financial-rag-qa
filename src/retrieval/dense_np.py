"""numpy 정확 검색 기반 Dense Retrieval (확장 브랜치, DECISIONS D-047).

`DenseRetriever`와 같은 입력·출력(`RetrievalResult`)을 쓰되 Qdrant 대신 `NumpyIndex`를 쓴다. Hybrid·Rerank·API는 검색 결과의
형식만 같으면 그대로 쓸 수 있다. 임베딩 모델·접두어·정규화는 기존과 같다. Langfuse 기록은 하지 않는다(확장은 로컬 결과만).
"""
from __future__ import annotations

import time

from src.index.embedding import Embedder
from src.index.numpy_store import NumpyIndex
from src.retrieval.dense import DenseRetriever, InvalidQuery, RetrievalConfig, RetrievalError, RetrievalResult


class NumpyDenseRetriever(DenseRetriever):
    def __init__(self, index: NumpyIndex, config: RetrievalConfig | None = None, *, embedder: Embedder | None = None):
        self.config = config or RetrievalConfig()
        self.embedder = embedder or Embedder(self.config.spec)
        self.index = index
        self.client = None
        self.langfuse = None

    def retrieve(self, question: str, k: int | None = None) -> RetrievalResult:
        return self._retrieve(question, k)

    def _retrieve(self, question: str, k: int | None) -> RetrievalResult:
        if not isinstance(question, str) or not question.strip():
            raise InvalidQuery("질문이 비어 있습니다.")
        k = self.config.k if k is None else k
        if k < 1:
            raise InvalidQuery("k는 1 이상이어야 합니다.")
        t0 = time.perf_counter()
        try:
            vector = self.embedder.embed_queries([question.strip()])[0]
            t1 = time.perf_counter()
            hits = self.index.search(vector, k=k)
            t2 = time.perf_counter()
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"{type(exc).__name__}: {str(exc)[:200]}") from exc
        chunks = [{"rank": i, "score": h["score"], "chunk_id": h["chunk_id"], "book_id": h["book_id"], "file_name": h["file_name"],
                   "page": h["page"], "text": h["chunk_text"], "metadata": h["metadata"]} for i, h in enumerate(hits, start=1)]
        return RetrievalResult(question=question, k=k, chunks=chunks,
                               durations_ms={"embed": round((t1 - t0) * 1000, 1), "search": round((t2 - t1) * 1000, 3),
                                             "total": round((t2 - t0) * 1000, 1)})
