"""Qdrant 접속과 컬렉션 관리 (STEP 3, DECISIONS.md D-010).

- `QDRANT_URL`과 `QDRANT_API_KEY`가 있으면 서버/Cloud, 없으면 `QDRANT_PATH`의 로컬 폴더를 쓴다.
- 컬렉션은 만들기만 하고 지우지 않는다. 이미 있으면 덮어쓰지 않고 오류를 낸다 (CLAUDE.md §4, 07_VERSIONING.md).
- 연결 오류는 짧게 재시도한다 (Cloud 첫 연결이 한 번 끊긴 사례가 있었다).
- 키 값은 출력하지 않는다.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Callable

from src.config import load_env

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parents[2]
_NAMESPACE = uuid.UUID("6f1d3c52-6a71-4a53-9f55-0d2d5b8f4c01")  # chunk_id -> 결정적 UUID


def point_id(chunk_id: str) -> str:
    """Qdrant 포인트 ID는 정수 또는 UUID여야 하므로 chunk_id에서 결정적으로 만든다."""
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def with_retry(fn: Callable[[], Any], attempts: int = 4, delay: float = 2.0) -> Any:
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - 네트워크 계열 오류만 재시도 대상이지만 종류가 다양하다
            last = exc
            logger.warning("Qdrant 호출 실패(%d/%d): %s", i + 1, attempts, type(exc).__name__)
            time.sleep(delay * (i + 1))
    raise last  # type: ignore[misc]


def qdrant_mode() -> str:
    load_env()
    return "remote" if os.environ.get("QDRANT_URL") else "local"


def get_client():
    """환경 변수에 따라 Qdrant 클라이언트를 만든다."""
    from qdrant_client import QdrantClient

    load_env()
    url = os.environ.get("QDRANT_URL", "").strip()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if url:
            return QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY", "").strip() or None, timeout=60)
        path = os.environ.get("QDRANT_PATH", "").strip() or str(_ROOT / "data" / "qdrant")
        return QdrantClient(path=path)


def collection_exists(client, name: str) -> bool:
    return with_retry(lambda: client.collection_exists(name))


def create_collection(client, name: str, dimension: int) -> None:
    """새 컬렉션을 만든다 (코사인 거리). 이미 있으면 ValueError."""
    from qdrant_client.models import Distance, VectorParams

    if collection_exists(client, name):
        raise ValueError(f"컬렉션이 이미 있습니다(덮어쓰지 않음): {name}")
    with_retry(lambda: client.create_collection(
        collection_name=name, vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
    ))


def chunk_payload(chunk: dict) -> dict:
    """색인에 저장하는 메타데이터. 평가 정보는 포함하지 않는다."""
    return {
        "chunk_id": chunk["chunk_id"], "book_id": chunk["book_id"], "file_name": chunk["file_name"],
        "page": chunk["page"], "chunk_index": chunk["chunk_index"], "chunk_text": chunk["chunk_text"],
        "metadata": chunk["metadata"], "chunking_version": chunk["chunking_version"],
        "dataset_version": chunk["dataset_version"],
    }


def upsert_chunks(client, name: str, chunks: list[dict], vectors: list[list[float]]) -> None:
    from qdrant_client.models import PointStruct

    points = [
        PointStruct(id=point_id(c["chunk_id"]), vector=v, payload=chunk_payload(c))
        for c, v in zip(chunks, vectors, strict=True)
    ]
    with_retry(lambda: client.upsert(collection_name=name, points=points, wait=True))


def search(client, name: str, vector: list[float], k: int = 5) -> list[dict]:
    """상위 k개 청크를 점수와 함께 돌려준다."""
    res = with_retry(lambda: client.query_points(
        collection_name=name, query=vector, limit=k, with_payload=True,
    ))
    return [{"score": p.score, **p.payload} for p in res.points]


def count_points(client, name: str) -> int:
    return with_retry(lambda: client.count(collection_name=name, exact=True)).count
