import pytest
from qdrant_client import QdrantClient

from src.index import qdrant_store as qs
from src.index.embedding import CANDIDATES, collection_name

CHUNK = {"chunk_id": "FB01-p1-c1", "book_id": "FB01", "file_name": "FB01.pdf", "page": 1, "chunk_index": 1,
         "chunk_text": "본문", "metadata": {}, "chunking_version": "chunk-v1", "dataset_version": "basic-v1",
         "expected_action": "answer", "reference_answer": "정답"}  # 평가용 필드가 섞여 있어도 저장되면 안 된다


def test_collection_name_includes_dataset_chunking_and_embedding_versions():
    assert collection_name("basic", "chunk-v1", CANDIDATES["e5"]) == "basic__chunk-v1__e5large-v1"
    assert collection_name("basic", "chunk-v1", CANDIDATES["bgem3"]) == "basic__chunk-v1__bgem3-v1"


def test_point_id_is_deterministic_uuid():
    assert qs.point_id("FB01-p1-c1") == qs.point_id("FB01-p1-c1") != qs.point_id("FB01-p1-c2")
    assert len(qs.point_id("x")) == 36


def test_payload_excludes_evaluation_fields():
    payload = qs.chunk_payload(CHUNK)
    assert "expected_action" not in payload and "reference_answer" not in payload
    assert {"chunk_id", "book_id", "file_name", "page"} <= set(payload)


def test_create_upsert_search_roundtrip_and_never_overwrites():
    client = QdrantClient(":memory:")
    qs.create_collection(client, "c", 3)
    with pytest.raises(ValueError):  # 이미 있으면 덮어쓰지 않고 오류
        qs.create_collection(client, "c", 3)
    other = {**CHUNK, "chunk_id": "FB01-p2-c1", "page": 2}
    qs.upsert_chunks(client, "c", [CHUNK, other], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert qs.count_points(client, "c") == 2
    hit = qs.search(client, "c", [0.0, 1.0, 0.0], k=1)[0]
    assert hit["chunk_id"] == "FB01-p2-c1" and hit["page"] == 2 and hit["file_name"] == "FB01.pdf"


def test_with_retry_retries_then_raises(monkeypatch):
    monkeypatch.setattr(qs.time, "sleep", lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("끊김")
        return "ok"

    assert qs.with_retry(flaky) == "ok" and len(calls) == 3
    with pytest.raises(ConnectionError):
        qs.with_retry(lambda: (_ for _ in ()).throw(ConnectionError("계속 실패")), attempts=2)
