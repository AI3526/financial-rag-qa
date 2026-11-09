"""FastAPI 앱 (STEP 7, 08 §11, 09 §8).

- `GET /health`: 서비스가 응답하는지만 확인한다.
- `POST /rag/query`: `{question, rag_method, question_id?}` → `{request_id, status, answer, reason, citations}`.
- 오류는 `status: "error"`와 `request_id`를 가진 같은 형식으로 반환한다(422/500/502/504).
- 선택 헤더: `traceparent`(평가 실험 Trace에 이어 붙기), `X-Experiment-Id`, `X-Track`, `X-Run-Type`.
  요청 본문 계약은 바꾸지 않는다. 응답 헤더 `X-Request-Id`, `X-Trace-Id`.
- Langfuse 기록과 run 레코드 저장이 실패해도 응답은 그대로 반환한다.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, Header, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.models import QueryRequest
from src.api.pipeline import GENERIC_ERROR_MESSAGE, Outcome, RagPipeline
from src.api.recording import EXPERIMENT_ID_RE, RunRecorder
from src.api.tracing import RequestTrace, request_trace

logger = logging.getLogger(__name__)

ALLOWED_TRACKS = {"development", "official", "extension"}
ALLOWED_RUN_TYPES = {"general_eval", "security", "mock_error", "scoring"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _error_response(request_id: str, reason: str) -> dict:
    return {"request_id": request_id, "status": "error", "answer": GENERIC_ERROR_MESSAGE, "reason": reason,
            "citations": []}


def _invalid_outcome(request_id: str, message: str) -> Outcome:
    record = {"retrieval": None, "final_context": None, "llm_calls": [], "model_input": None,
              "llm_output_raw": None, "llm_output_parsed": None,
              "error": {"reason": "invalid_request", "error_stage": "validate", "error_message": message[:300],
                        "retry_count": 0, "is_retryable": False},
              "performance": {"total_duration_ms": 0.0, "retrieval_duration_ms": None, "generation_duration_ms": None,
                              "retry_count": 0, "final_status": "error"}}
    return Outcome(422, _error_response(request_id, "invalid_request"), record)


def create_app(pipeline: RagPipeline, *, langfuse=None, recorder: RunRecorder | None = None,
               default_track: str = "development") -> FastAPI:
    app = FastAPI(title="Financial RAG QA")

    def versions(outcome: Outcome) -> dict:
        cfg = getattr(pipeline.retriever, "config", None)
        model_id = next((c["model_id"] for c in outcome.record.get("llm_calls", []) if c.get("model_id")), None)
        return {"dataset_version": getattr(cfg, "dataset_version", None),
                "chunking_version": getattr(cfg, "chunking_version", None),
                "embedding_version": getattr(getattr(cfg, "spec", None), "version_tag", None),
                "model_id": model_id, "config_version": None, "prompt_version": None}

    def finalize(*, request_id: str, outcome: Outcome, received_at: str, trace: RequestTrace, run: dict,
                 question, rag_method, question_id) -> JSONResponse:
        record = {
            "request_id": request_id, "trace_id": trace.trace_id, "experiment_id": run["experiment_id"],
            "track": run["track"], "run_type": run["run_type"], "question_id": question_id,
            "evaluation_set_type": None, "rag_method": rag_method, "question": question,
            "versions": versions(outcome), **outcome.record, "response": outcome.response,
            "http_status": outcome.http_status,
            "observability": {"langfuse_recorded": trace.recorded, "langfuse_error": trace.error},
            "received_at": received_at, "completed_at": _now(),
        }
        if recorder is not None:
            recorder.write(run["experiment_id"], record)
        headers = {"X-Request-Id": request_id}
        if trace.trace_id:
            headers["X-Trace-Id"] = trace.trace_id
        return JSONResponse(status_code=outcome.http_status, content=outcome.response, headers=headers)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/rag/query")
    def query(body: QueryRequest,
              traceparent: str | None = Header(default=None),
              x_experiment_id: str | None = Header(default=None),
              x_track: str | None = Header(default=None),
              x_run_type: str | None = Header(default=None)) -> JSONResponse:
        request_id, received_at = str(uuid.uuid4()), _now()
        run = {"experiment_id": x_experiment_id, "track": x_track or default_track,
               "run_type": x_run_type or "general_eval"}
        header_error = None
        if run["track"] not in ALLOWED_TRACKS:
            header_error = "올바르지 않은 X-Track"
        elif run["run_type"] not in ALLOWED_RUN_TYPES:
            header_error = "올바르지 않은 X-Run-Type"
        elif x_experiment_id is not None and not EXPERIMENT_ID_RE.match(x_experiment_id):
            header_error = "올바르지 않은 X-Experiment-Id"
        if header_error:
            run["experiment_id"] = None  # 검증되지 않은 값은 저장 경로에 쓰지 않는다
            return finalize(request_id=request_id, outcome=_invalid_outcome(request_id, header_error),
                            received_at=received_at, trace=RequestTrace(), run=run, question=body.question,
                            rag_method=body.rag_method, question_id=body.question_id)

        with request_trace(
            langfuse, traceparent=traceparent, environment=run["track"], tags=[run["run_type"]],
            input={"question": body.question, "rag_method": body.rag_method, "question_id": body.question_id},
            metadata={"request_id": request_id, "question_id": body.question_id,
                      "experiment_id": run["experiment_id"], "track": run["track"], "run_type": run["run_type"],
                      "rag_method": body.rag_method},
        ) as trace:
            outcome = pipeline.run(request_id, body.question, body.rag_method)
            trace.finish(output=outcome.response, failed=outcome.response["status"] == "error",
                         status_message=outcome.response["reason"])
        return finalize(request_id=request_id, outcome=outcome, received_at=received_at, trace=trace, run=run,
                        question=body.question, rag_method=body.rag_method, question_id=body.question_id)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """필드 누락·자료형 오류도 같은 오류 형식(422 invalid_request)으로 돌려준다."""
        request_id, received_at = str(uuid.uuid4()), _now()
        fields = ", ".join(sorted({str(e["loc"][-1]) for e in exc.errors() if e.get("loc")}))
        return finalize(request_id=request_id, outcome=_invalid_outcome(request_id, f"요청 형식 오류: {fields}"),
                        received_at=received_at, trace=RequestTrace(),
                        run={"experiment_id": None, "track": default_track, "run_type": "general_eval"},
                        question=None, rag_method=None, question_id=None)

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException):
        """본문을 읽을 수 없는 요청(예: UTF-8이 아닌 바이트)은 FastAPI 기본 400 대신 공통 오류 형식(422)으로 돌려준다.
        그 밖의 HTTP 오류(404, 405 등)는 질문 처리 계약 밖이므로 기본 응답을 유지한다."""
        if exc.status_code == 400:
            request_id = str(uuid.uuid4())
            return finalize(request_id=request_id, outcome=_invalid_outcome(request_id, "요청 본문을 읽을 수 없습니다"),
                            received_at=_now(), trace=RequestTrace(),
                            run={"experiment_id": None, "track": default_track, "run_type": "general_eval"},
                            question=None, rag_method=None, question_id=None)
        return await http_exception_handler(request, exc)

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """파이프라인 밖에서 난 예상하지 못한 오류. 내부 내용은 응답에 담지 않는다."""
        logger.exception("처리되지 않은 오류")
        request_id = str(uuid.uuid4())
        return JSONResponse(status_code=500, content=_error_response(request_id, "internal_error"),
                            headers={"X-Request-Id": request_id})

    return app
