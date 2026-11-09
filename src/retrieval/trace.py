"""검색 호출을 Langfuse Span으로 감싸는 공통 도구 (08 §5 계측 원칙).

Span이므로 요청 Trace 안에서 호출되면 자동으로 그 아래에 붙고, 혼자 실행하면 자기 Trace를 만든다(D-029).
Langfuse 오류는 모두 삼키며 검색 결과와 예외는 그대로 전달한다.
"""
from __future__ import annotations

import logging
from contextlib import ExitStack
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _safe(fn: Callable[[], Any], what: str) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse %s 실패: %s", what, type(exc).__name__)


def traced_retrieval(langfuse, name: str, span_input: dict, metadata: dict, run: Callable[[], Any],
                     output_of: Callable[[Any], dict]) -> Any:
    """run()을 실행하고 결과를 Span `name`(유형 retriever)에 기록한다. langfuse가 None이면 그냥 run()."""
    if langfuse is None:
        return run()
    stack = ExitStack()
    try:
        span = stack.enter_context(langfuse.start_as_current_observation(
            name=name, as_type="retriever", input=span_input, metadata=metadata))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse Span 시작 실패: %s", type(exc).__name__)
        stack.close()
        return run()
    try:
        result = run()
    except Exception as exc:
        _safe(lambda: span.update(level="ERROR", status_message=f"{type(exc).__name__}: {str(exc)[:200]}"), "Span 기록")
        _safe(stack.close, "Span 종료")
        raise
    _safe(lambda: span.update(output=output_of(result)), "Span 기록")
    _safe(stack.close, "Span 종료")
    return result
