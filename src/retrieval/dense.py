"""Dense Retrieval (STEP 4, 08_DEVELOPMENT_PLAN.md §8).

질문을 동결된 임베딩 모델(DECISIONS.md D-027, e5-large)로 벡터화하고 Qdrant에서 Top-k 청크를 찾는다.

- 반환 형식은 이후 Hybrid·Rerank와 API가 공통으로 쓴다: rank(1부터), score, 청크 본문과 출처(book_id, file_name, page).
- 유사도 임계값은 두지 않는다(설계에 없음). 근거가 충분한지는 생성 단계가 판단하고, 검색은 항상 Top-k를 돌려준다.
- 평가 정답·근거 원문은 이 모듈에 들어오지 않는다. 입력은 질문 문자열뿐이다.
- Langfuse 기록(08 §5)은 선택이다. `langfuse`를 주면 검색마다 Span을 기록한다. Span이므로 요청 Trace 안에서 호출되면
  자동으로 그 아래에 붙고, 요청 Trace의 ID 설계(STEP 7 전 spike)와 무관하다. 단계별 시간은 결과에도 담는다.
"""
from __future__ import annotations

import logging
import time
from contextlib import ExitStack
from dataclasses import dataclass, field

from src.index import qdrant_store as qs
from src.index.embedding import CANDIDATES, Embedder, EmbeddingSpec, collection_name


logger = logging.getLogger(__name__)


class InvalidQuery(ValueError):
    """빈 질문 등 잘못된 입력. API에서는 422 invalid_request로 연결한다."""


class RetrievalError(RuntimeError):
    """검색 저장소·임베딩 실패. API에서는 500 retrieval_error로 연결한다."""


@dataclass(frozen=True)
class RetrievalConfig:
    """공식 Dense 검색 설정. 동결 후 변경은 새 실험 ID로 한다."""

    dataset_id: str = "basic"
    dataset_version: str = "basic-v1"
    chunking_version: str = "chunk-v1"
    embedding: str = "e5"  # D-027 동결
    k: int = 5

    @property
    def spec(self) -> EmbeddingSpec:
        return CANDIDATES[self.embedding]

    @property
    def collection(self) -> str:
        return collection_name(self.dataset_id, self.chunking_version, self.spec)


@dataclass
class RetrievalResult:
    question: str
    method: str = "dense"
    k: int = 5
    chunks: list[dict] = field(default_factory=list)  # [{rank, score, chunk_id, book_id, file_name, page, text, metadata}]
    durations_ms: dict = field(default_factory=dict)  # {"embed", "search", "total"}


class DenseRetriever:
    def __init__(self, config: RetrievalConfig | None = None, *, embedder: Embedder | None = None, client=None,
                 langfuse=None):
        """langfuse: 주어지면 검색마다 Span `dense-retrieve`를 기록한다(08 §5). 현재 열린 Trace가 있으면 그 아래에
        붙고 없으면 자기 Trace를 만든다. 기록 실패는 검색을 막지 않는다."""
        self.config = config or RetrievalConfig()
        self.embedder = embedder or Embedder(self.config.spec)
        self.client = client or qs.get_client()
        self.langfuse = langfuse

    def retrieve(self, question: str, k: int | None = None) -> RetrievalResult:
        if self.langfuse is None:
            return self._retrieve(question, k)
        return self._retrieve_traced(question, k)

    def _retrieve_traced(self, question: str, k: int | None) -> RetrievalResult:
        span_stack = ExitStack()
        span = None
        try:
            span = span_stack.enter_context(self.langfuse.start_as_current_observation(
                name="dense-retrieve", as_type="retriever",
                input={"question": question, "k": self.config.k if k is None else k},
                metadata={"method": "dense", "collection": self.config.collection,
                          "embedding_model": self.config.spec.hf_id, "embedding_version": self.config.spec.version_tag},
            ))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse Span 시작 실패: %s", type(exc).__name__)
            span_stack.close()
            return self._retrieve(question, k)
        try:
            result = self._retrieve(question, k)
        except Exception as exc:
            self._safe_update(span, level="ERROR", status_message=f"{type(exc).__name__}: {str(exc)[:200]}")
            self._safe_close(span_stack)
            raise
        self._safe_update(span, output={
            "chunks": [{"rank": c["rank"], "score": round(c["score"], 4), "chunk_id": c["chunk_id"],
                        "book_id": c["book_id"], "file_name": c["file_name"], "page": c["page"], "text": c["text"]}
                       for c in result.chunks],
            "durations_ms": result.durations_ms,
        })
        self._safe_close(span_stack)
        return result

    @staticmethod
    def _safe_update(span, **kwargs) -> None:
        try:
            span.update(**kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse Span 기록 실패: %s", type(exc).__name__)

    @staticmethod
    def _safe_close(stack: ExitStack) -> None:
        try:
            stack.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse Span 종료 실패: %s", type(exc).__name__)

    def _retrieve(self, question: str, k: int | None) -> RetrievalResult:
        if not isinstance(question, str) or not question.strip():
            raise InvalidQuery("질문이 비어 있습니다.")
        k = self.config.k if k is None else k  # `k or 기본값`은 k=0을 기본값으로 바꿔 버린다
        if k < 1:
            raise InvalidQuery("k는 1 이상이어야 합니다.")
        t0 = time.perf_counter()
        try:
            vector = self.embedder.embed_queries([question.strip()])[0]
            t1 = time.perf_counter()
            hits = qs.search(self.client, self.config.collection, vector, k=k)
            t2 = time.perf_counter()
        except Exception as exc:  # noqa: BLE001 - 원인 종류와 무관하게 retrieval_error로 통일한다
            raise RetrievalError(f"{type(exc).__name__}: {str(exc)[:200]}") from exc
        chunks = [
            {"rank": i, "score": h["score"], "chunk_id": h["chunk_id"], "book_id": h["book_id"],
             "file_name": h["file_name"], "page": h["page"], "text": h["chunk_text"], "metadata": h["metadata"]}
            for i, h in enumerate(hits, start=1)
        ]
        return RetrievalResult(
            question=question, k=k, chunks=chunks,
            durations_ms={"embed": round((t1 - t0) * 1000, 1), "search": round((t2 - t1) * 1000, 1),
                          "total": round((t2 - t0) * 1000, 1)},
        )
