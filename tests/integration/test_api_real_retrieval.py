"""STEP 7: 실제 검색(Qdrant Cloud + e5)을 API에 붙여 확인한다. 생성기만 가짜다(STEP 8 전).

네트워크와 모델이 필요하다. 컬렉션에 접속할 수 없으면 건너뛴다.
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


def _available() -> bool:
    if qs.qdrant_mode() != "remote":
        return False
    try:
        return qs.collection_exists(qs.get_client(), RetrievalConfig().collection)
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _available(), reason="Qdrant 컬렉션에 접속할 수 없습니다")


class CitingGenerator:
    """검색된 첫 출처(S1)를 그대로 근거로 삼는 가짜 생성기."""

    def __init__(self):
        self.seen = []

    def generate(self, question, sources):
        self.seen.append((question, sources))
        return GenerationOutput(parsed={"status": "answered", "answer": "테스트 답변", "reason": "S1 근거",
                                        "source_ids": ["S1"]}, model_id="fake")


@pytest.fixture(scope="module")
def retriever(dense_retriever):
    return dense_retriever


def test_real_retrieval_through_api_maps_citation_to_real_pdf_page(retriever, tmp_path):
    gen = CitingGenerator()
    client = TestClient(create_app(RagPipeline(retriever, gen, PipelineSettings()), recorder=RunRecorder(tmp_path)))
    r = client.post("/rag/query", json={"question": "2026년 상반기 영업이익은 얼마인가요?", "rag_method": "dense"})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "answered"
    cite = body["citations"][0]
    top = gen.seen[0][1][0]
    assert cite == {"source_id": "S1", "book_id": top["book_id"], "file_name": top["file_name"], "page": top["page"]}
    assert cite["file_name"].endswith(".pdf") and cite["page"] >= 1
    rec = json.loads((tmp_path / "service" / "runs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert len(rec["retrieval"]["dense"]) == 5 and rec["retrieval"]["durations_ms"]["total"] > 0
    assert rec["versions"]["embedding_version"] == "e5large-v1" and rec["versions"]["dataset_version"] == "basic-v1"


def test_real_retrieval_passes_only_the_question_to_generation(retriever, tmp_path):
    gen = CitingGenerator()
    client = TestClient(create_app(RagPipeline(retriever, gen, PipelineSettings())))
    client.post("/rag/query", json={"question": "루미센서 E는 어떤 제품인가요?", "rag_method": "dense",
                                    "reference_answer": "유출되면 안 되는 정답", "expected_action": "answer"})
    flat = json.dumps(gen.seen, ensure_ascii=False)
    assert "유출되면 안 되는 정답" not in flat and gen.seen[0][0] == "루미센서 E는 어떤 제품인가요?"
