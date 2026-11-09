"""STEP 8: 실제 검색(Qdrant, e5) + 실제 gpt-6-luna 생성기를 API로 연결해 확인한다.

OpenAI 호출이 소량 발생한다(정상 3회 + 모델명 오류 1회, timeout 테스트는 서버에 도달하기 전에 실패). 키와 컬렉션이 없으면
건너뛴다. 응답은 매번 달라질 수 있어 특정 `status`를 단언하지 않고 형식·출처 유효성을 확인한다.
"""
import json
import os

import pytest
from fastapi.testclient import TestClient

from src.config import load_env

load_env()

from src.api.app import create_app  # noqa: E402
from src.api.pipeline import PipelineSettings, RagPipeline  # noqa: E402
from src.api.recording import RunRecorder  # noqa: E402
from src.generation.generator import LangChainGenerator  # noqa: E402
from src.index import qdrant_store as qs  # noqa: E402
from src.retrieval.dense import DenseRetriever, RetrievalConfig  # noqa: E402


def _available() -> bool:
    if not (os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_MODEL")) or qs.qdrant_mode() != "remote":
        return False
    try:
        return qs.collection_exists(qs.get_client(), RetrievalConfig().collection)
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _available(), reason="OpenAI 키 또는 Qdrant 컬렉션을 사용할 수 없습니다")


@pytest.fixture(scope="module")
def retriever():
    return DenseRetriever()


def client_for(retriever, tmp_path, settings=None, **gen_kwargs):
    settings = settings or PipelineSettings()
    pipe = RagPipeline(retriever, LangChainGenerator(settings, **gen_kwargs), settings, sleep=lambda s: None)
    return TestClient(create_app(pipe, recorder=RunRecorder(tmp_path)), raise_server_exceptions=False)


def last_record(tmp_path):
    return json.loads((tmp_path / "service" / "runs.jsonl").read_text(encoding="utf-8").splitlines()[-1])


@pytest.mark.parametrize("question", [
    "2026년 상반기 영업이익은 얼마인가요?",  # 근거 있는 질문
    "해온솔루션의 2027년 1분기 확정 매출을 알려 주세요.",  # 자료에 없는 질문
    "리브의 재생 원료 비율은 얼마인가요?",  # 모호한 질문
])
def test_real_generation_returns_valid_status_and_traceable_citations(retriever, tmp_path, question):
    r = client_for(retriever, tmp_path).post("/rag/query", json={"question": question, "rag_method": "dense"})
    body = r.json()
    assert r.status_code == 200, body
    assert body["status"] in ("answered", "needs_clarification", "abstained") and body["answer"].strip()
    rec = last_record(tmp_path)
    table = {s["source_id"]: s for s in rec["final_context"]["chunks"]}
    for c in body["citations"]:  # 모든 인용이 이번 요청의 최종 문맥과 실제 PDF·페이지에 대응한다
        s = table[c["source_id"]]
        assert (c["book_id"], c["file_name"], c["page"]) == (s["book_id"], s["file_name"], s["page"])
        assert c["file_name"].endswith(".pdf") and c["page"] >= 1
    call = rec["llm_calls"][0]
    assert isinstance(call["input_tokens"], int) and isinstance(call["output_tokens"], int)
    assert call["model_id"] and call["duration_ms"] > 0 and call["status"] == "ok"
    assert rec["versions"]["prompt_version"] == "answer-v2" and len(rec["versions"]["prompt_sha256"]) == 64
    assert rec["model_input"][1]["content"].startswith("## 검색 자료") and question in rec["model_input"][1]["content"]
    assert rec["final_context"]["context_limit_tokens"] == 4000 and rec["final_context"]["dropped_chunk_ids"] == []


def test_real_generation_never_receives_evaluation_fields(retriever, tmp_path):
    r = client_for(retriever, tmp_path).post("/rag/query", json={
        "question": "루미센서 E는 어떤 제품인가요?", "rag_method": "dense",
        "reference_answer": "유출되면 안 되는 정답", "expected_action": "answer"})
    assert r.status_code == 200
    assert "유출되면 안 되는 정답" not in json.dumps(last_record(tmp_path), ensure_ascii=False)


def test_tiny_timeout_returns_504_model_timeout(retriever, tmp_path):
    settings = PipelineSettings(model_timeout_seconds=0.001)
    r = client_for(retriever, tmp_path, settings).post("/rag/query", json={"question": "영업이익은?", "rag_method": "dense"})
    assert r.status_code == 504 and r.json()["reason"] == "model_timeout"
    assert len(last_record(tmp_path)["llm_calls"]) == 3  # 재시도 2회까지 합쳐 3회


def test_unknown_model_name_returns_502_without_retry(retriever, tmp_path):
    r = client_for(retriever, tmp_path, model="no-such-model-xyz").post(
        "/rag/query", json={"question": "영업이익은?", "rag_method": "dense"})
    assert r.status_code == 502 and r.json()["reason"] == "model_error"
    assert len(last_record(tmp_path)["llm_calls"]) == 1  # 재시도해도 같으므로 1회로 끝난다
