"""STEP 6: 실제 Reranker(bge-reranker-v2-m3)와 실제 Hybrid(Qdrant + BM25)를 확인한다. 생성기는 가짜(OpenAI 호출 없음).

모델(약 2.3GB)과 Qdrant 컬렉션이 필요하다. 없으면 건너뛴다.
"""
import json

import pytest
from fastapi.testclient import TestClient

from src.config import load_env

load_env()

from src.api.app import create_app  # noqa: E402
from src.api.pipeline import GenerationOutput, PipelineSettings, RagPipeline  # noqa: E402
from src.api.recording import RunRecorder  # noqa: E402
from src.index import qdrant_store as qs  # noqa: E402
from src.index.embedding import cached_revision  # noqa: E402
from src.retrieval.dense import DenseRetriever, RetrievalConfig  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402
from src.retrieval.rerank import RERANKER_ID, CrossEncoderReranker, RerankRetriever  # noqa: E402
from src.retrieval.sparse import CHUNKS_PATH, SparseRetriever  # noqa: E402


def _available() -> bool:
    if qs.qdrant_mode() != "remote" or not CHUNKS_PATH.exists() or cached_revision(RERANKER_ID) is None:
        return False
    try:
        return qs.collection_exists(qs.get_client(), RetrievalConfig().collection)
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _available(), reason="Reranker 모델·Qdrant 컬렉션·청크 파일 중 하나가 없습니다")


@pytest.fixture(scope="module")
def reranker(shared_reranker):
    return shared_reranker


@pytest.fixture(scope="module")
def rerank_retriever(reranker, dense_retriever, sparse_retriever):
    return RerankRetriever(HybridRetriever(dense_retriever, sparse_retriever), reranker)


def test_model_identity_revision_and_device(reranker):
    assert reranker.model_id == "BAAI/bge-reranker-v2-m3"
    assert reranker.revision and len(reranker.revision) == 40  # 확인된 revision(커밋 해시)이 기록된다


def test_relevant_chunk_scores_higher_than_unrelated_chunk(reranker):
    q = "루미센서 E는 어떤 제품인가요?"
    good, bad = reranker.score(q, ["루미센서 E는 해온솔루션 장비사업의 확장형 제품이다.", "오늘 저녁 메뉴는 김치찌개다."])
    assert good > bad and 0.0 <= bad <= good <= 1.0  # Sigmoid 점수


def test_scores_are_deterministic_and_empty_input_is_handled(reranker):
    texts = ["영업이익은 96억원이다.", "해온소재의 필름사업 매출은 620억원이다."]
    assert reranker.score("영업이익은?", texts) == reranker.score("영업이익은?", texts)
    assert reranker.score("영업이익은?", []) == []


def test_hybrid_rerank_returns_final_five_with_before_after_for_twenty_candidates(rerank_retriever):
    res = rerank_retriever.retrieve("2026년 상반기 영업이익은 얼마인가요?")
    assert res.method == "hybrid_rerank" and len(res.chunks) == 5
    assert [c["rank"] for c in res.chunks] == [1, 2, 3, 4, 5]
    scores = [c["score"] for c in res.chunks]
    assert scores == sorted(scores, reverse=True)
    rr = res.candidates["reranked"]
    assert len(rr) == 20 and sorted(r["rank_before"] for r in rr) == list(range(1, 21))
    assert [r["rank_after"] for r in rr] == list(range(1, 21))
    assert [c["chunk_id"] for c in res.chunks] == [r["chunk_id"] for r in rr[:5]]
    assert all(c["file_name"] == c["book_id"] + ".pdf" and c["page"] >= 1 for c in res.chunks)  # 출처가 보존된다
    assert set(res.durations_ms) >= {"dense", "sparse", "fusion", "rerank", "total"}


def test_rerank_does_not_add_chunks_outside_the_hybrid_candidates(rerank_retriever):
    res = rerank_retriever.retrieve("루미센서 E는 어떤 제품인가요?")
    fused_ids = {c["chunk_id"] for c in res.candidates["fused"]}
    assert {c["chunk_id"] for c in res.chunks} <= fused_ids  # 재정렬은 후보 안에서만 고른다


class CitingGenerator:
    def __init__(self):
        self.seen = []

    def generate(self, question, sources):
        self.seen.append((question, sources))
        return GenerationOutput(parsed={"status": "answered", "answer": "테스트", "reason": "S1", "source_ids": ["S1"]}, model_id="fake")


def test_api_hybrid_rerank_end_to_end_with_same_final_k_and_no_evaluation_leak(rerank_retriever, tmp_path):
    gen = CitingGenerator()
    pipe = RagPipeline(rerank_retriever, gen, PipelineSettings(), retrievers={"hybrid_rerank": rerank_retriever})
    client = TestClient(create_app(pipe, recorder=RunRecorder(tmp_path)))
    r = client.post("/rag/query", json={"question": "루미센서 E는 어떤 제품인가요?", "rag_method": "hybrid_rerank",
                                        "reference_answer": "유출되면 안 되는 정답"})
    assert r.status_code == 200 and r.json()["citations"][0]["file_name"].endswith(".pdf")
    rec = json.loads((tmp_path / "service" / "runs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert len(rec["final_context"]["chunks"]) == 5  # 세 방식이 같은 최종 k를 쓴다
    assert len(rec["retrieval"]["reranked"]) == 20 and rec["retrieval"]["settings"]["reranker_revision"]
    assert rec["performance"]["reranking_duration_ms"] > 0
    assert "유출되면 안 되는 정답" not in json.dumps(gen.seen, ensure_ascii=False)
