"""Langfuse 공통 초기화와 요청 식별자 생성.

- request_id: 서버가 요청마다 만드는 UUID (프로젝트 내부 식별자)
- trace_id: request_id에서 결정적으로 만든 Langfuse Trace ID (같은 request_id -> 같은 trace_id)

Langfuse 설정이 없거나 연결에 문제가 있어도 답변 생성과 결과 저장이 막히면 안 되므로
(04_ARCHITECTURE.md 13.4) 이 모듈은 예외를 밖으로 던지지 않고 None을 돌려준다.
키 값은 로그와 출력에 남기지 않는다.
"""
from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_REQUIRED_ENV = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


@dataclass(frozen=True)
class RequestIds:
    request_id: str
    trace_id: str


def trace_id_for(request_id: str) -> str:
    """request_id에서 Langfuse Trace ID(32자리 소문자 16진수)를 결정적으로 만든다."""
    from langfuse import Langfuse

    return Langfuse.create_trace_id(seed=request_id)


def new_request_ids(request_id: str | None = None) -> RequestIds:
    """새 요청의 식별자 쌍을 만든다. request_id를 주면 그 값을 쓴다(재현·테스트용)."""
    rid = request_id or str(uuid.uuid4())
    return RequestIds(request_id=rid, trace_id=trace_id_for(rid))


def langfuse_configured() -> bool:
    """필수 환경 변수가 모두 설정되어 있는지 (값은 확인만 하고 출력하지 않는다)."""
    load_dotenv(_ROOT / ".env")
    return all(os.environ.get(name) for name in _REQUIRED_ENV)


def get_langfuse():
    """Langfuse 클라이언트를 돌려준다. 설정이 없거나 초기화에 실패하면 None."""
    if not langfuse_configured():
        logger.warning("Langfuse 환경 변수가 설정되지 않아 관측 기록을 건너뜁니다.")
        return None
    try:
        from langfuse import get_client

        return get_client()
    except Exception as exc:  # noqa: BLE001 - 관측 실패가 서비스를 막지 않게 한다
        logger.warning("Langfuse 초기화 실패: %s", type(exc).__name__)
        return None
