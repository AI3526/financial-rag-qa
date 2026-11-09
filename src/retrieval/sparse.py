"""BM25 Sparse Retrieval (STEP 5, 08 §9, 06 §5.3).

같은 `chunks.jsonl`을 메모리에서 색인한다(토크나이저는 `whitespace`로 동결, D-035)(청크가 1천 개 수준이라 충분). 입력 텍스트는 Dense와 같은 검색용 텍스트
(`retrieval_text`: 머리말·문서 제목·소제목 + 본문)이고, 질문과 청크에 같은 토크나이저를 쓴다.

- 질문의 토큰이 하나도 없는 청크는 결과에 넣지 않는다. 그래서 결과가 k개보다 적거나 비어 있을 수 있고, 이는 오류가 아니다
  (Hybrid에서는 Dense 결과만 남는다). 일치 여부는 점수가 아니라 토큰 일치로 판단한다: rank_bm25는 질문 단어가 전체 청크의
  절반 이상에 나오면 IDF가 0 이하가 되어 단어가 일치해도 점수가 0일 수 있기 때문이다.
- 같은 점수는 chunk_index(문서·페이지 순서)로 정렬해 결과가 항상 같다.
- 평가 정답·근거 원문은 이 모듈에 들어오지 않는다. 입력은 청크 파일과 질문 문자열뿐이다.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from src.config import ROOT
from src.ingest import chunk as ck
from src.retrieval.dense import InvalidQuery, RetrievalError, RetrievalResult, result_output
from src.retrieval.tokenize import get_tokenizer
from src.retrieval.trace import traced_retrieval

CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class SparseRetriever:
    def __init__(self, chunks: list[dict] | None = None, tokenizer_name: str = "whitespace", *,
                 tokenizer: Callable[[str], list[str]] | None = None, k: int = 5, langfuse=None):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks if chunks is not None else load_chunks()
        self.tokenizer_name = tokenizer_name if tokenizer is None else "custom"
        self.tokenize = tokenizer or get_tokenizer(tokenizer_name)
        self.k = k
        self.langfuse = langfuse
        t0 = time.perf_counter()
        self.index_seconds: float | None = None
        corpus = [self.tokenize(ck.retrieval_text(c)) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus) if any(corpus) else None
        self.index_seconds = round(time.perf_counter() - t0, 3)

    def retrieve(self, question: str, k: int | None = None) -> RetrievalResult:
        return traced_retrieval(
            self.langfuse, "sparse-retrieve", {"question": question, "k": self.k if k is None else k},
            {"method": "sparse", "algorithm": "bm25", "tokenizer": self.tokenizer_name,
             "chunk_count": len(self.chunks)},
            lambda: self._retrieve(question, k), result_output)

    def _retrieve(self, question: str, k: int | None) -> RetrievalResult:
        if not isinstance(question, str) or not question.strip():
            raise InvalidQuery("질문이 비어 있습니다.")
        k = self.k if k is None else k
        if k < 1:
            raise InvalidQuery("k는 1 이상이어야 합니다.")
        t0 = time.perf_counter()
        try:
            tokens = self.tokenize(question.strip())
            scores = self.bm25.get_scores(tokens) if (self.bm25 is not None and tokens) else []
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"{type(exc).__name__}: {str(exc)[:200]}") from exc
        t1 = time.perf_counter()
        wanted = set(tokens)
        matched = [i for i in range(len(self.chunks)) if wanted & self.bm25.doc_freqs[i].keys()] if len(scores) else []
        ranked = sorted(matched, key=lambda i: (-scores[i], self.chunks[i]["chunk_index"], i))[:k]
        chunks = [
            {"rank": r, "score": float(scores[i]), "chunk_id": self.chunks[i]["chunk_id"], "book_id": self.chunks[i]["book_id"],
             "file_name": self.chunks[i]["file_name"], "page": self.chunks[i]["page"], "text": self.chunks[i]["chunk_text"],
             "metadata": self.chunks[i]["metadata"]}
            for r, i in enumerate(ranked, start=1)
        ]
        return RetrievalResult(question=question, method="sparse", k=k, chunks=chunks,
                               durations_ms={"search": round((t1 - t0) * 1000, 1), "total": round((t1 - t0) * 1000, 1)})
