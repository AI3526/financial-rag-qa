"""요청 Trace (STEP 7, DECISIONS.md D-029).

- `traceparent` 헤더(W3C 형식 `00-{trace_id 32자}-{span_id 16자}-{flags}`)가 있으면 그 Trace 아래에 이어 붙는다
  (평가 실험이 만든 Trace). 이때 `environment`는 건드리지 않는다(실험 Trace는 SDK가 고정).
- 헤더가 없으면 새 Trace(`rag-query`)를 만들고 `environment = track`을 쓴다.
- Langfuse 오류는 모두 삼킨다. 기록 실패가 응답과 결과 저장을 막지 않는다 (04_ARCHITECTURE.md 13.4).
"""
from __future__ import annotations

import logging
import re
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_TRACEPARENT = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}$")


def parse_traceparent(value: str | None) -> dict | None:
    """올바른 `traceparent`면 {trace_id, parent_span_id}, 아니면 None(무시). 모두 0인 ID는 무효다."""
    if not value:
        return None
    m = _TRACEPARENT.match(value.strip())
    if not m or set(m.group(1)) == {"0"} or set(m.group(2)) == {"0"}:
        return None
    return {"trace_id": m.group(1), "parent_span_id": m.group(2)}


@dataclass
class RequestTrace:
    trace_id: str | None = None
    continued: bool = False  # 전달받은 Trace에 이어 붙었는가
    error: str | None = None  # Langfuse 기록 실패 사유(비밀 제외)
    _span: object = field(default=None, repr=False)

    @property
    def recorded(self) -> bool:
        return self._span is not None and self.error is None

    def finish(self, *, output: dict, failed: bool = False, status_message: str | None = None) -> None:
        if self._span is None:
            return
        try:
            self._span.update(output=output, level="ERROR" if failed else "DEFAULT",
                              status_message=status_message if failed else None)
        except Exception as exc:  # noqa: BLE001
            self.error = type(exc).__name__
            logger.warning("Langfuse 요청 Span 기록 실패: %s", self.error)


@contextmanager
def request_trace(langfuse, *, traceparent: str | None, input: dict, tags: list[str], metadata: dict,
                  environment: str):
    state = RequestTrace()
    if langfuse is None:
        yield state
        return
    stack = ExitStack()
    try:
        from langfuse import propagate_attributes

        meta = {k: str(v) for k, v in metadata.items() if v is not None}
        ctx = parse_traceparent(traceparent)
        if ctx:
            stack.enter_context(propagate_attributes(tags=tags, metadata=meta))
            span = stack.enter_context(langfuse.start_as_current_observation(
                trace_context=ctx, name="rag-query", as_type="span", input=input))
            state.continued = True
        else:
            stack.enter_context(propagate_attributes(trace_name="rag-query", environment=environment,
                                                     tags=tags, metadata=meta))
            span = stack.enter_context(langfuse.start_as_current_observation(
                name="rag-query", as_type="span", input=input))
        state._span = span
        state.trace_id = getattr(span, "trace_id", None) or (ctx or {}).get("trace_id")
    except Exception as exc:  # noqa: BLE001
        state.error = type(exc).__name__
        logger.warning("Langfuse 요청 Trace 시작 실패: %s", state.error)
        try:
            stack.close()
        except Exception:  # noqa: BLE001
            pass
        stack = ExitStack()
    try:
        yield state
    finally:
        try:
            stack.close()
        except Exception as exc:  # noqa: BLE001
            state.error = state.error or type(exc).__name__
            logger.warning("Langfuse 요청 Trace 종료 실패: %s", type(exc).__name__)
