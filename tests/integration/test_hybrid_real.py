"""STEP 5: 실제 데이터(chunks.jsonl)와 실제 Qdrant·e5로 BM25와 Hybrid를 확인한다. 생성기는 가짜(OpenAI 호출 없음).

BM25는 로컬 파일만 필요하고, Hybrid와 API는 Qdrant 컬렉션이 필요하다(없으면 건너뛴다).
질문은 평가 데이터가 아니라 문서의 이름·용어로 직접 만든 것이다.
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
from src.retrieval.dense import DenseRetriever, RetrievalConfig  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402
from src.retrieval.sparse import CHUNKS_PATH, SparseRetriever  # noqa: E402


def _qdrant_ok() -> bool:
    if qs.qdrant_mode() != "remote":
        return False
    try:
        return qs.collection_exists(qs.get_client(), RetrievalConfig().collection)
    except Exception:  # noqa: BLE001
        return False


needs_qdrant = pytest.mark.skipif(not _qdrant_ok(), reason="Qdrant 컬렉션에 접속할 수 없습니다")
needs_chunks = pytest.mark.skipif(not CHUNKS_PATH.exists(), reason="chunks.jsonl이 없습니다")

TYPED = [
    ("회사명", "해온소재는 어떤 회사인가요?", "해온소재"),
    ("제품명", "루미센서 E는 어떤 제품인가요?", "루미센서"),
    ("제품명2", "리브필름 기본형의 재생 원료 비율은?", "리브필름"),
    # BM25 한계: "상반기"는 청크의 절반에 나와 IDF가 거의 0이고 "주세요"는 1개에만 있어 요청 문구가 순위를 좌우한다(D-036)
    ("기간", "2026년 상반기 실적을 알려 주세요.", None),
    ("수치", "영업이익은 얼마인가요?", "영업이익"),
    ("사업부문", "장비사업과 운영서비스사업은 어떻게 구분되나요?", "운영서비스"),
]


@pytest.fixture(scope="module")
def sparse(sparse_retriever):
    return sparse_retriever


@needs_chunks
@pytest.mark.parametrize("kind,question,term", TYPED, ids=[t[0] for t in TYPED])
def test_bm25_returns_ranked_chunks_containing_the_query_term(sparse, kind, question, term):
    res = sparse.retrieve(question)
    assert res.method == "sparse" and 1 <= len(res.chunks) <= 5
    assert [c["rank"] for c in res.chunks] == list(range(1, len(res.chunks) + 1))
    scores = [c["score"] for c in res.chunks]
    assert scores == sorted(scores, reverse=True)
    assert all(c["file_name"] == c["book_id"] + ".pdf" and c["page"] >= 1 and c["text"].strip() for c in res.chunks)
    if term is not None:
        assert any(term in c["text"] for c in res.chunks), f"{kind}: BM25 상위 청크에 '{term}'이 없음"


@needs_chunks
def test_bm25_is_fast_and_deterministic(sparse):
    a, b = sparse.retrieve("루미뷰 이용계약은 어떻게 되나요?"), sparse.retrieve("루미뷰 이용계약은 어떻게 되나요?")
    assert [c["chunk_id"] for c in a.chunks] == [c["chunk_id"] for c in b.chunks]
    assert a.durations_ms["total"] < 500
    assert sparse.retrieve("김치찌개 레시피").chunks == [] or len(sparse.retrieve("김치찌개 레시피").chunks) <= 5  # 오류 없이 처리


@needs_chunks
@needs_qdrant
def test_hybrid_returns_final_five_with_candidates_and_traceable_sources(sparse, dense_retriever):
    res = HybridRetriever(dense_retriever, sparse).retrieve("2026년 상반기 영업이익은 얼마인가요?")
    assert res.method == "hybrid" and len(res.chunks) == 5 and res.candidate_k == 20
    assert [c["rank"] for c in res.chunks] == [1, 2, 3, 4, 5]
    assert len(res.candidates["dense"]) == 20 and len(res.candidates["fused"]) == 20
    assert len(res.candidates["sparse"]) <= 20 and set(res.durations_ms) == {"dense", "sparse", "fusion", "total"}
    assert all(c["file_name"] == c["book_id"] + ".pdf" and c["page"] >= 1 for c in res.chunks)
    fused_ids = [c["chunk_id"] for c in res.candidates["fused"]]
    assert [c["chunk_id"] for c in res.chunks] == fused_ids[:5]  # 최종 5개는 결합 결과의 앞 5개다


class CitingGenerator:
    def __init__(self):
        self.seen = []

    def generate(self, question, sources):
        self.seen.append((question, sources))
        return GenerationOutput(parsed={"status": "answered", "answer": "테스트", "reason": "S1", "source_ids": ["S1"]},
                                model_id="fake")


@needs_chunks
@needs_qdrant
def test_api_hybrid_uses_the_same_k_and_records_dense_sparse_and_fused(sparse, dense_retriever, tmp_path):
    dense = dense_retriever
    gen = CitingGenerator()
    pipe = RagPipeline(dense, gen, PipelineSettings(), retrievers={"dense": dense, "hybrid": HybridRetriever(dense, sparse)})
    client = TestClient(create_app(pipe, recorder=RunRecorder(tmp_path)))
    for method in ("dense", "hybrid"):
        r = client.post("/rag/query", json={"question": "루미센서 E는 어떤 제품인가요?", "rag_method": method,
                                            "reference_answer": "유출되면 안 되는 정답"})
        assert r.status_code == 200 and r.json()["citations"][0]["file_name"].endswith(".pdf")
    recs = [json.loads(l) for l in (tmp_path / "service" / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [len(r["final_context"]["chunks"]) for r in recs] == [5, 5]  # 세 방식이 같은 최종 k를 쓴다
    assert recs[0]["retrieval"]["sparse"] is None and recs[1]["retrieval"]["fused"]
    assert recs[1]["retrieval"]["settings"]["bm25_tokenizer"] == "whitespace" and recs[1]["retrieval"]["settings"]["rrf_k"] == 60
    assert "유출되면 안 되는 정답" not in json.dumps(gen.seen, ensure_ascii=False)
