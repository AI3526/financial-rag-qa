import json

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.pipeline import GenerationOutput, PipelineSettings, RagPipeline
from src.retrieval.dense import InvalidQuery, RetrievalError, RetrievalResult
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.rerank import RerankConfig, RerankRetriever, rerank_chunks
from tests.unit.test_api import BrokenLangfuse, FakeLangfuse


def chunk(rank, cid, text=None, meta=None, book="FB01", page=1):
    return {"rank": rank, "score": 1 / (60 + rank), "chunk_id": cid, "book_id": book, "file_name": f"{book}.pdf",
            "page": page, "text": text or f"본문 {cid}", "metadata": meta or {}}


class FakeHybrid:
    def __init__(self, chunks, error=None):
        self.chunks, self.error, self.calls = chunks, error, []

    def retrieve(self, question, k=5):
        self.calls.append((question, k))
        if self.error:
            raise self.error
        cands = {"dense": [{"chunk_id": c["chunk_id"], "rank": c["rank"]} for c in self.chunks], "sparse": [], "fused": []}
        return RetrievalResult(question=question, method="hybrid", k=k, chunks=self.chunks[:k], candidates=cands,
                               candidate_k=20, durations_ms={"dense": 5.0, "sparse": 1.0, "fusion": 0.1, "total": 6.1})

    def describe(self):
        return {"fusion": "rrf", "rrf_k": 60}


class FakeReranker:
    """텍스트에 들어 있는 `[점수=…]` 표시를 그대로 점수로 돌려주는 가짜 모델."""

    model_id, revision = "fake-reranker", "abc123"

    def __init__(self, scores=None, error=None):
        self.scores, self.error, self.calls = scores, error, []

    def score(self, question, texts):
        self.calls.append((question, texts))
        if self.error:
            raise self.error
        return self.scores if self.scores is not None else [0.0] * len(texts)


def candidates(n=20):
    return [chunk(i, f"C{i}") for i in range(1, n + 1)]


# --- 재정렬 ---
def test_rerank_orders_by_score_and_records_rank_before_and_after():
    chunks = candidates(4)
    out = rerank_chunks("질문", chunks, lambda q, t: [0.1, 0.9, 0.5, 0.7])
    assert [c["chunk_id"] for c in out] == ["C2", "C4", "C3", "C1"]
    assert [(c["rank_before"], c["rank_after"], c["rank"]) for c in out] == [(2, 1, 1), (4, 2, 2), (3, 3, 3), (1, 4, 4)]
    assert out[0]["score"] == 0.9 and out[0]["rerank_score"] == 0.9
    assert (out[0]["book_id"], out[0]["file_name"], out[0]["page"], out[0]["text"]) == ("FB01", "FB01.pdf", 1, "본문 C2")


def test_ties_keep_the_earlier_rank_so_results_are_deterministic():
    out = rerank_chunks("질문", candidates(4), lambda q, t: [0.5, 0.5, 0.5, 0.5])
    assert [c["chunk_id"] for c in out] == ["C1", "C2", "C3", "C4"]


def test_reranker_input_is_the_same_retrieval_text_with_title_context_and_never_evaluation_data():
    meta = {"header": "FB01 | 머리말", "doc_title": "문서 제목", "section_title": "소제목"}
    seen = []
    rerank_chunks("질문?", [chunk(1, "C1", "본문입니다", meta)], lambda q, t: seen.append((q, t)) or [1.0])
    q, texts = seen[0]
    assert q == "질문?" and texts == ["FB01 | 머리말 / 문서 제목 / 소제목\n본문입니다"]  # 임베딩·BM25와 같은 입력 텍스트


def test_empty_candidates_do_not_call_the_model():
    called = []
    assert rerank_chunks("질문", [], lambda q, t: called.append(1) or []) == [] and called == []


def test_score_count_mismatch_is_a_retrieval_error():
    with pytest.raises(RetrievalError):
        rerank_chunks("질문", candidates(3), lambda q, t: [0.1])


# --- RerankRetriever ---
def retriever(scores=None, hybrid=None, reranker=None, **kw):
    return RerankRetriever(hybrid or FakeHybrid(candidates()), reranker or FakeReranker(scores), **kw)


def test_uses_the_initial_settings_twenty_candidates_to_final_five():
    h, r = FakeHybrid(candidates(20)), FakeReranker([float(i) for i in range(20)])  # 20번 후보가 최고점
    res = RerankRetriever(h, r).retrieve("질문")
    assert h.calls == [("질문", 20)] and len(r.calls[0][1]) == 20
    assert res.method == "hybrid_rerank" and res.k == 5 and len(res.chunks) == 5
    assert [c["chunk_id"] for c in res.chunks] == ["C20", "C19", "C18", "C17", "C16"]
    assert [c["rank"] for c in res.chunks] == [1, 2, 3, 4, 5]
    assert RerankConfig() == RerankConfig(20, 5)


def test_records_before_after_for_all_candidates_and_keeps_the_hybrid_candidates():
    res = retriever([float(i) for i in range(20)]).retrieve("질문")
    rr = res.candidates["reranked"]
    assert len(rr) == 20 and set(rr[0]) == {"chunk_id", "rank_before", "rank_after", "score"}
    assert rr[0] == {"chunk_id": "C20", "rank_before": 20, "rank_after": 1, "score": 19.0}
    assert "dense" in res.candidates and set(res.durations_ms) >= {"dense", "fusion", "rerank", "hybrid_total", "total"}


def test_final_k_comes_from_the_argument_and_blank_input_is_invalid():
    res = retriever([float(i) for i in range(20)]).retrieve("질문", k=3)
    assert len(res.chunks) == 3
    r = retriever()
    for bad in ("", "  ", None):
        with pytest.raises(InvalidQuery):
            r.retrieve(bad)
    with pytest.raises(InvalidQuery):
        r.retrieve("질문", k=0)


def test_hybrid_errors_propagate_and_reranker_errors_become_retrieval_errors():
    with pytest.raises(RetrievalError):
        retriever(hybrid=FakeHybrid([], error=RetrievalError("qdrant down"))).retrieve("질문")
    with pytest.raises(RetrievalError) as e:
        retriever(reranker=FakeReranker(error=RuntimeError("CUDA out of memory"))).retrieve("질문")
    assert "Reranker" in str(e.value)


def test_no_candidates_returns_empty_result_without_error():
    res = RerankRetriever(FakeHybrid([]), FakeReranker()).retrieve("질문")
    assert res.chunks == [] and res.candidates["reranked"] == []


def test_describe_records_reranker_model_and_revision():
    d = retriever().describe()
    assert d["reranker"] == "fake-reranker" and d["reranker_revision"] == "abc123" and d["rerank_candidates"] == 20
    assert d["rrf_k"] == 60 and d["final_k"] == 5


def test_langfuse_span_and_failure_isolation():
    lf = FakeLangfuse()
    retriever([float(i) for i in range(20)], langfuse=lf).retrieve("질문")
    assert lf.spans[0].name == "rerank-retrieve" and lf.spans[0].kwargs["metadata"]["reranker"] == "fake-reranker"
    out = lf.spans[0].updates[0]["output"]
    assert len(out["candidates"]["reranked"]) == 20 and len(out["chunks"]) == 5
    assert len(retriever([float(i) for i in range(20)], langfuse=BrokenLangfuse()).retrieve("질문").chunks) == 5


def test_real_hybrid_retriever_can_feed_the_reranker():
    class D:
        def retrieve(self, q, k=5):
            return RetrievalResult(question=q, k=k, chunks=[chunk(i, f"D{i}") for i in range(1, 4)])

    class S:
        tokenizer_name = "whitespace"

        def retrieve(self, q, k=5):
            return RetrievalResult(question=q, method="sparse", k=k, chunks=[chunk(1, "D2")])

    # RRF 결합 순서는 D2(양쪽에 있음), D1, D3다. 점수 [0.1, 0.9, 0.5]는 이 순서의 후보에 하나씩 붙는다.
    res = RerankRetriever(HybridRetriever(D(), S()), FakeReranker([0.1, 0.9, 0.5])).retrieve("질문")
    assert [c["chunk_id"] for c in res.chunks] == ["D1", "D3", "D2"]
    assert [(c["chunk_id"], c["rank_before"]) for c in res.chunks] == [("D1", 2), ("D3", 3), ("D2", 1)]


# --- API 연결 ---
class Gen:
    def generate(self, question, sources):
        return GenerationOutput(parsed={"status": "answered", "answer": "답", "reason": "r", "source_ids": ["S1"]}, model_id="fake")


def test_api_hybrid_rerank_records_reranked_list_and_rerank_duration(tmp_path):
    from src.api.recording import RunRecorder

    rr = RerankRetriever(FakeHybrid(candidates()), FakeReranker([float(i) for i in range(20)]))
    pipe = RagPipeline(rr, Gen(), PipelineSettings(), retrievers={"hybrid_rerank": rr})
    client = TestClient(create_app(pipe, recorder=RunRecorder(tmp_path)))
    r = client.post("/rag/query", json={"question": "질문", "rag_method": "hybrid_rerank"})
    assert r.status_code == 200 and r.json()["citations"][0]["source_id"] == "S1"
    rec = json.loads((tmp_path / "service" / "runs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rec["retrieval"]["method"] == "hybrid_rerank" and len(rec["retrieval"]["reranked"]) == 20
    assert rec["retrieval"]["durations_ms"]["rerank"] >= 0 and rec["performance"]["reranking_duration_ms"] is not None
    assert [c["chunk_id"] for c in rec["final_context"]["chunks"]] == ["C20", "C19", "C18", "C17", "C16"]
    assert rec["retrieval"]["settings"]["reranker"] == "fake-reranker"


def test_api_reranker_failure_returns_500_retrieval_error_and_does_not_fall_back():
    rr = RerankRetriever(FakeHybrid(candidates()), FakeReranker(error=RuntimeError("boom")))
    pipe = RagPipeline(rr, Gen(), PipelineSettings(), retrievers={"hybrid_rerank": rr})
    r = TestClient(create_app(pipe), raise_server_exceptions=False).post("/rag/query", json={"question": "질문", "rag_method": "hybrid_rerank"})
    assert r.status_code == 500 and r.json()["reason"] == "retrieval_error" and r.json()["status"] == "error"
