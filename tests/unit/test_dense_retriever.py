import pytest
from qdrant_client import QdrantClient

from src.index import qdrant_store as qs
from src.retrieval.dense import DenseRetriever, InvalidQuery, RetrievalConfig, RetrievalError


class FakeEmbedder:
    """질문 문자열에 따라 정해진 벡터를 돌려주는 가짜 임베더 (모델 로드 없이 검색 로직만 확인)."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def embed_queries(self, texts):
        self.calls += texts
        return [self.mapping[t] for t in texts]


def _chunk(i, book, page, text):
    return {"chunk_id": f"{book}-p{page}-c{i}", "book_id": book, "file_name": f"{book}.pdf", "page": page,
            "chunk_index": i, "chunk_text": text, "metadata": {"header": f"{book} | x"},
            "chunking_version": "chunk-v1", "dataset_version": "basic-v1"}


@pytest.fixture()
def retriever():
    client = QdrantClient(":memory:")
    cfg = RetrievalConfig()
    qs.create_collection(client, cfg.collection, 3)
    chunks = [_chunk(1, "FB01", 1, "가"), _chunk(1, "FB02", 2, "나"), _chunk(1, "FB03", 3, "다"), _chunk(1, "FB04", 4, "라")]
    vectors = [[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0, 1]]
    qs.upsert_chunks(client, cfg.collection, chunks, vectors)
    emb = FakeEmbedder({"질문": [1, 0, 0], "다른 질문": [0, 0, 1]})
    return DenseRetriever(cfg, embedder=emb, client=client)


def test_returns_ranked_chunks_with_source_fields_and_scores(retriever):
    res = retriever.retrieve("질문", k=3)
    assert res.method == "dense" and res.k == 3 and len(res.chunks) == 3
    assert [c["rank"] for c in res.chunks] == [1, 2, 3]
    assert res.chunks[0]["chunk_id"] == "FB01-p1-c1" and res.chunks[1]["chunk_id"] == "FB02-p2-c1"
    scores = [c["score"] for c in res.chunks]
    assert scores == sorted(scores, reverse=True)
    top = res.chunks[0]
    assert {"book_id", "file_name", "page", "text", "metadata", "score", "rank", "chunk_id"} <= set(top)
    assert (top["file_name"], top["page"]) == ("FB01.pdf", 1)  # 원본 PDF 페이지로 역추적
    assert set(res.durations_ms) == {"embed", "search", "total"}


def test_default_k_comes_from_config_and_question_is_stripped(retriever):
    res = retriever.retrieve("  질문  ")
    assert len(res.chunks) == 4  # 청크가 4개뿐이라 k=5여도 4개
    assert retriever.embedder.calls == ["질문"]


def test_blank_question_is_invalid_and_never_searches(retriever):
    for bad in ("", "   ", "\n\t", None):
        with pytest.raises(InvalidQuery):
            retriever.retrieve(bad)
    with pytest.raises(InvalidQuery):
        retriever.retrieve("질문", k=0)
    assert retriever.embedder.calls == []


def test_empty_collection_returns_empty_list_not_error():
    client = QdrantClient(":memory:")
    cfg = RetrievalConfig()
    qs.create_collection(client, cfg.collection, 3)
    r = DenseRetriever(cfg, embedder=FakeEmbedder({"질문": [1, 0, 0]}), client=client)
    res = r.retrieve("질문")
    assert res.chunks == [] and res.k == 5


def test_storage_failure_becomes_retrieval_error(monkeypatch, retriever):
    monkeypatch.setattr(qs.time, "sleep", lambda s: None)

    def boom(*a, **k):
        raise ConnectionError("저장소 연결 실패")

    monkeypatch.setattr(retriever.client, "query_points", boom)
    with pytest.raises(RetrievalError):
        retriever.retrieve("질문")


def _traced(retriever, lf):
    retriever.langfuse = lf
    return retriever


def test_records_dense_retrieve_span_with_question_results_and_timing(retriever):
    from tests.unit.test_ingestion_trace import StubLangfuse

    lf = StubLangfuse()
    res = _traced(retriever, lf).retrieve("질문", k=2)
    assert [s.name for s in lf.spans] == ["dense-retrieve"]
    span = lf.spans[0]
    assert span.kwargs["as_type"] == "retriever"
    assert span.kwargs["input"] == {"question": "질문", "k": 2}
    assert span.kwargs["metadata"]["embedding_version"] == "e5large-v1"
    out = span.updates[0]["output"]
    assert [c["chunk_id"] for c in out["chunks"]] == [c["chunk_id"] for c in res.chunks]
    assert {"rank", "score", "file_name", "page", "text"} <= set(out["chunks"][0])
    assert set(out["durations_ms"]) == {"embed", "search", "total"}


def test_failed_retrieval_is_marked_error_and_still_raises(retriever):
    from tests.unit.test_ingestion_trace import StubLangfuse

    lf = StubLangfuse()
    with pytest.raises(InvalidQuery):
        _traced(retriever, lf).retrieve("   ")
    assert lf.spans[0].updates[0]["level"] == "ERROR"


def test_langfuse_failure_never_blocks_retrieval(retriever):
    from tests.unit.test_ingestion_trace import BrokenLangfuse

    res = _traced(retriever, BrokenLangfuse()).retrieve("질문", k=2)
    assert len(res.chunks) == 2


def test_retrieval_does_not_use_evaluation_fields(retriever):
    res = retriever.retrieve("질문")
    flat = str(res.chunks)
    assert "expected_action" not in flat and "reference_answer" not in flat
