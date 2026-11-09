"""평가 LLM 호출의 Langfuse 기록 (STEP 11, 05_OBSERVABILITY_PLAN.md §3.4, D-038).

서비스 답변 생성(`rag-query` Trace의 `answer-generation`)과 **구분**한다.
- 별도 Trace, 이름 `faithfulness-evaluation` 또는 `judge-evaluation`, 유형 Generation
- 태그 `run_type:scoring`, `track:…`, `rag_method:…` (일반 지표 집계에서 제외하는 구분)
- metadata에 평가 대상 요청의 `request_id`·`trace_id`를 남겨 서비스 Trace와 연결한다(서비스 Trace 아래에 두지 않는다)
Langfuse 오류는 삼키고 채점 결과에 영향을 주지 않는다.
"""
from __future__ import annotations

import logging
from contextlib import ExitStack

logger = logging.getLogger(__name__)


def record_scoring(langfuse, *, name: str, model: str | None, track: str, rag_method: str, evaluated: dict, input: dict,
                   output: dict, usage: dict | None, metadata: dict) -> str | None:
    """평가 호출 1건을 기록하고 Trace ID를 돌려준다. langfuse가 None이거나 실패하면 None."""
    if langfuse is None:
        return None
    stack = ExitStack()
    try:
        from langfuse import propagate_attributes

        md = {k: str(v) for k, v in {**metadata, "evaluated_request_id": evaluated.get("request_id"),
                                     "evaluated_trace_id": evaluated.get("trace_id"),
                                     "question_id": evaluated.get("question_id"), "run_type": "scoring",
                                     "track": track, "rag_method": rag_method}.items() if v is not None}
        stack.enter_context(propagate_attributes(
            trace_name=name, environment=track,
            tags=["run_type:scoring", f"track:{track}", f"rag_method:{rag_method}"], metadata=md))
        gen = stack.enter_context(langfuse.start_as_current_observation(
            name=name, as_type="generation", model=model, input=input, output=output,
            usage_details=({"input": usage["input_tokens"], "output": usage["output_tokens"]} if usage else None),
            metadata=metadata))
        return getattr(gen, "trace_id", None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse 평가 기록 실패: %s", type(exc).__name__)
        return None
    finally:
        try:
            stack.close()
        except Exception:  # noqa: BLE001
            pass
