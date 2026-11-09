"""질문 하나를 처리하는 파이프라인과 오류 구분 (STEP 7, 08 §11, 09 §8).

검증 → 검색 → 생성(교체 가능한 Generator) → 출력 검증 → 출처 변환 → 응답.
- 생성기는 이 단계에서 인터페이스만 둔다. 실제 gpt-6-luna 연결은 STEP 8이다. 생성기가 연결되지 않은 서버는
  답을 만들어 내지 않고 `internal_error`를 돌려준다.
- 오류 구분: 입력 검증 422, 검색 500, 모델 502/504, 출력 검증 502, 내부 500. LLM은 `error`를 판단하지 않는다.
- 평가 정답·근거는 이 코드에 전달되지 않는다. 생성기와 검색은 질문 문자열과 청크만 받는다.
- timeout·재시도 값은 DECISIONS.md D-030의 초기값이다(공식 실행 전 확정).
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol

from src.generation.context import TOKEN_COUNT_METHOD, build_final_context
from src.generation.pricing import request_cost
from src.retrieval.dense import InvalidQuery, RetrievalError
from src.retrieval.trace import traced_retrieval

logger = logging.getLogger(__name__)

ANSWER_STATUSES = ("answered", "needs_clarification", "abstained")
GENERIC_ERROR_MESSAGE = "요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요."


# --- 오류 ---------------------------------------------------------------------------------------------
class PipelineError(Exception):
    """API 응답으로 변환되는 오류. reason은 09 §8.5의 코드다."""

    reason = "internal_error"
    http_status = 500
    retryable = False

    def __init__(self, message: str = "", *, retryable: bool | None = None):
        super().__init__(message)
        self.message = message
        if retryable is not None:  # 같은 오류 종류라도 원인에 따라 재시도 가능 여부가 다르다(예: 429 vs 401)
            self.retryable = retryable


class InvalidRequest(PipelineError):
    reason, http_status = "invalid_request", 422


class RetrievalFailure(PipelineError):
    reason, http_status = "retrieval_error", 500


class ModelError(PipelineError):
    reason, http_status, retryable = "model_error", 502, True


class ModelTimeout(PipelineError):
    reason, http_status, retryable = "model_timeout", 504, True


class OutputValidationError(PipelineError):
    reason, http_status = "output_validation_error", 502


class GenerationUnavailable(PipelineError):
    """생성기가 연결되지 않음(STEP 8 전). 재시도해도 해결되지 않는다."""

    reason, http_status = "internal_error", 500


# --- 설정 (D-030 초기값) -------------------------------------------------------------------------------
@dataclass(frozen=True)
class PipelineSettings:
    final_k: int = 5
    model_timeout_seconds: float = 30.0  # 생성기가 호출 때 적용한다
    max_model_calls: int = 3  # 재시도와 재생성의 합
    max_transient_retries: int = 2  # 연결 오류·429·5xx·timeout
    retry_delays_seconds: tuple[float, ...] = (1.0, 2.0)
    max_regenerations: int = 1  # 출력 검증 실패 시 다시 생성
    # 생성 설정 (D-032 제안값, 세 RAG 방식에 동일 적용, 공식 실행 전 development로 확인 후 동결)
    reasoning_effort: str = "low"
    max_completion_tokens: int = 1500
    seed: int = 20261109
    context_limit_tokens: int = 4000  # 최종 문맥 한도(추정 토큰, context.py)


# --- 생성기 인터페이스 ----------------------------------------------------------------------------------
@dataclass
class GenerationOutput:
    parsed: Any  # LLM이 반환한 JSON 객체(검증 전)
    raw: str | None = None
    model_input: Any = None
    model_id: str | None = None
    usage: dict | None = None  # input_tokens, output_tokens, reasoning_tokens (확인 불가는 None)


class Generator(Protocol):
    def generate(self, question: str, sources: list[dict]) -> GenerationOutput: ...


class UnavailableGenerator:
    """STEP 8 전의 기본 생성기. 답을 만들지 않고 오류를 낸다."""

    def generate(self, question: str, sources: list[dict]) -> GenerationOutput:
        raise GenerationUnavailable("생성 모델이 연결되지 않았습니다(STEP 8 전).")


# --- 출력 검증 (출처 ID 대응표 생성은 src/generation/context.py) -------------------------------------------
def validate_output(parsed: Any, sources: list[dict]) -> dict:
    """LLM 출력을 검증한다. 통과하면 {status, answer, reason, source_ids}를 돌려주고, 아니면 OutputValidationError."""
    if not isinstance(parsed, dict):
        raise OutputValidationError("출력이 JSON 객체가 아닙니다.")
    for key in ("status", "answer", "reason", "source_ids"):
        if key not in parsed:
            raise OutputValidationError(f"필수 필드 누락: {key}")
    if parsed["status"] not in ANSWER_STATUSES:  # LLM은 `error`를 반환할 수 없다
        raise OutputValidationError(f"허용되지 않는 status: {parsed['status']!r}")
    if not isinstance(parsed["answer"], str) or not isinstance(parsed["reason"], str):
        raise OutputValidationError("answer와 reason은 문자열이어야 합니다.")
    ids = parsed["source_ids"]
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise OutputValidationError("source_ids는 문자열 배열이어야 합니다.")
    known = {s["source_id"] for s in sources}
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise OutputValidationError(f"이번 요청의 문맥에 없는 출처 ID: {unknown}")
    return {"status": parsed["status"], "answer": parsed["answer"], "reason": parsed["reason"],
            "source_ids": list(dict.fromkeys(ids))}


def citations_for(source_ids: list[str], sources: list[dict]) -> list[dict]:
    table = {s["source_id"]: s for s in sources}
    return [{"source_id": i, "book_id": table[i]["book_id"], "file_name": table[i]["file_name"],
             "page": table[i]["page"]} for i in source_ids]


# --- 파이프라인 ----------------------------------------------------------------------------------------
@dataclass
class Outcome:
    http_status: int
    response: dict
    record: dict = field(default_factory=dict)  # run 레코드에 들어갈 내용(09 §9)


class RagPipeline:
    def __init__(self, retriever, generator: Generator | None = None, settings: PipelineSettings | None = None,
                 *, sleep: Callable[[float], None] = time.sleep, pricing=None, retrievers: dict | None = None,
                 langfuse=None):
        # 방식별 검색기(`rag_method` → 검색기). 주어지지 않으면 Dense 하나다. hybrid_rerank는 STEP 6에서 추가한다.
        self.retrievers = dict(retrievers) if retrievers else {"dense": retriever}
        self.retriever = retriever if retriever is not None else self.retrievers.get("dense")
        self.generator = generator or UnavailableGenerator()
        self.settings = settings or PipelineSettings()
        self._sleep = sleep
        self.pricing = pricing  # configs/pricing.json의 단가. 없으면 비용을 계산하지 않는다(null)
        self.langfuse = langfuse  # 있으면 문맥 구성·출처 변환 단계를 Span으로 기록한다(요청 Trace 안에서 호출될 때 하위에 붙음)

    def version_info(self, rag_method: str | None = None) -> dict:
        """이 파이프라인의 데이터셋·청킹·임베딩·프롬프트·모델 버전과 `config_version`(실행 설정 스냅샷의 해시).

        config_version = `cfg-` + sha256(데이터셋·청킹·임베딩·프롬프트(sha256)·모델·파이프라인 설정·선택한 검색 방식의 설정)의 앞 10자리.
        같은 설정이면 같은 값이고 하나라도 바뀌면 달라진다(D-038).
        """
        cfg = getattr(self.retriever, "config", None)
        gen = self.generator
        info = {"dataset_id": getattr(cfg, "dataset_id", None), "dataset_version": getattr(cfg, "dataset_version", None),
                "chunking_version": getattr(cfg, "chunking_version", None),
                "embedding_version": getattr(getattr(cfg, "spec", None), "version_tag", None),
                "prompt_version": getattr(gen, "prompt_version", None), "prompt_sha256": getattr(gen, "prompt_sha256", None),
                "model": getattr(getattr(gen, "llm", None), "model_name", None)}
        retriever = self.retrievers.get(rag_method) if rag_method else None
        snapshot = {**info, "settings": asdict(self.settings),
                    "retrieval": retriever.describe() if hasattr(retriever, "describe") else None}
        digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        return {**info, "config_version": f"cfg-{digest[:10]}"}

    # 입력 검증은 API 계층에서도 하지만, 파이프라인을 직접 부르는 평가 코드를 위해 여기서도 막는다
    def validate_request(self, question: Any, rag_method: Any) -> str:
        if not isinstance(question, str) or not question.strip():
            raise InvalidRequest("질문이 비어 있습니다.")
        if rag_method not in self.retrievers:
            raise InvalidRequest(f"지원하지 않는 rag_method: {rag_method!r}")
        return question.strip()

    def run(self, request_id: str, question: Any, rag_method: Any) -> Outcome:
        started = time.perf_counter()
        record: dict = {"retrieval": None, "final_context": None, "llm_calls": [], "model_input": None,
                        "llm_output_raw": None, "llm_output_parsed": None, "error": None}
        sources: list[dict] = []
        stage = "validate"
        try:
            question = self.validate_request(question, rag_method)
            stage = "retrieval"
            sources = self._retrieve(question, rag_method, record)
            stage = "generation"
            parsed = self._generate(question, sources, record)
            response = traced_retrieval(
                self.langfuse, "citation-and-status",
                {"llm_status": parsed["status"], "llm_source_ids": parsed["source_ids"],
                 "context_source_ids": [x["source_id"] for x in sources]}, {},
                lambda: {"request_id": request_id, "status": parsed["status"], "answer": parsed["answer"],
                         "reason": parsed["reason"], "citations": citations_for(parsed["source_ids"], sources)},
                # 이 단계에 도달했다는 것은 출력 검증(상태 값, 필수 필드, 출처 ID가 문맥에 있음)을 통과했다는 뜻이다
                lambda r: {"validation": "passed", "status": r["status"], "reason": r["reason"], "citations": r["citations"]},
                as_type="span")
            record["llm_output_parsed"] = parsed
            outcome = Outcome(200, response, record)
        except PipelineError as exc:
            outcome = self._error_outcome(request_id, exc, stage, record)
        except Exception as exc:  # noqa: BLE001 - 예상하지 못한 오류도 형식을 지켜 반환한다
            logger.exception("예상하지 못한 오류")
            outcome = self._error_outcome(request_id, PipelineError(f"{type(exc).__name__}"), stage, record)
        record.update(request_cost(record["llm_calls"], self.pricing))
        record["performance"] = {
            "total_duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "retrieval_duration_ms": (record["retrieval"] or {}).get("durations_ms", {}).get("total"),
            "reranking_duration_ms": (record["retrieval"] or {}).get("durations_ms", {}).get("rerank"),
            "generation_duration_ms": round(sum(c.get("duration_ms") or 0 for c in record["llm_calls"]), 1) or None,
            "retry_count": max(0, len(record["llm_calls"]) - 1),
            "final_status": outcome.response["status"],
        }
        return outcome

    # --- 단계 ---
    def _retrieve(self, question: str, rag_method: str, record: dict) -> list[dict]:
        retriever = self.retrievers[rag_method]
        try:
            result = retriever.retrieve(question, k=self.settings.final_k)
        except InvalidQuery as exc:
            raise InvalidRequest(str(exc)) from exc
        except RetrievalError as exc:
            raise RetrievalFailure(str(exc)) from exc
        context = traced_retrieval(
            self.langfuse, "context-construction",
            {"retrieved": [{"rank": c["rank"], "chunk_id": c["chunk_id"], "book_id": c["book_id"], "page": c["page"]}
                           for c in result.chunks], "context_limit_tokens": self.settings.context_limit_tokens},
            {"token_count_method": TOKEN_COUNT_METHOD},
            lambda: build_final_context(result.chunks, self.settings.context_limit_tokens),
            lambda c: {"retrieved_count": len(result.chunks), "final_count": len(c.sources),
                       "context_token_count": c.token_count, "context_limit_tokens": c.limit_tokens,
                       "token_count_method": TOKEN_COUNT_METHOD, "dropped_chunk_ids": c.dropped_chunk_ids,
                       "over_limit": c.over_limit,
                       # 모델에 실제 전달한 문맥(검색 결과와 구분). 본문은 Generation 입력에 있다
                       "final_chunks": [{"source_id": s["source_id"], "chunk_id": s["chunk_id"], "book_id": s["book_id"],
                                         "file_name": s["file_name"], "page": s["page"], "rank": s["rank"]}
                                        for s in c.sources]},
            as_type="span")
        brief = [{"chunk_id": c["chunk_id"], "book_id": c["book_id"], "page": c["page"], "rank": c["rank"],
                  "score": c["score"]} for c in result.chunks]
        cand = result.candidates or {}
        record["retrieval"] = {
            # 사용하지 않은 방식은 null이다(0이나 빈 배열로 바꾸지 않는다). 점수는 방식마다 의미가 달라 단순 비교하지 않는다.
            "method": rag_method, "dense": cand.get("dense", brief if rag_method == "dense" else None),
            "sparse": cand.get("sparse"), "fused": cand.get("fused"), "reranked": cand.get("reranked"),
            "candidate_k": result.candidate_k or result.k, "final_k": result.k, "durations_ms": result.durations_ms,
            "settings": retriever.describe() if hasattr(retriever, "describe") else None,
        }
        record["final_context"] = context.as_record()  # 1차 검색 결과(retrieval)와 구분: 모델에 실제 전달한 문맥
        return context.sources

    def _generate(self, question: str, sources: list[dict], record: dict) -> dict:
        s = self.settings
        transient = regenerations = 0
        while True:
            index = len(record["llm_calls"])
            started = time.perf_counter()
            call = {"call_index": index, "stage": "generation", "model_id": None, "retry_index": index,
                    "input_tokens": None, "output_tokens": None, "reasoning_tokens": None,
                    "cached_input_tokens": None, "cache_write_tokens": None, "service_tier": None, "duration_ms": None,
                    "status": None, "usage_status": "unknown"}
            record["llm_calls"].append(call)
            try:
                out = self.generator.generate(question, sources)
                call["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
                call["model_id"] = out.model_id
                if out.usage:
                    call.update(input_tokens=out.usage.get("input_tokens"), output_tokens=out.usage.get("output_tokens"),
                                reasoning_tokens=out.usage.get("reasoning_tokens"),
                                cached_input_tokens=out.usage.get("cached_input_tokens"),
                                cache_write_tokens=out.usage.get("cache_write_tokens"),
                                service_tier=out.usage.get("service_tier"), usage_status="reported")
                record["model_input"], record["llm_output_raw"] = out.model_input, out.raw
                parsed = validate_output(out.parsed, sources)
                call["status"] = "ok"
                return parsed
            except GenerationUnavailable:
                record["llm_calls"].pop()  # 모델을 호출하지 않았으므로 호출 기록에 남기지 않는다
                raise
            except PipelineError as exc:
                call["duration_ms"] = call["duration_ms"] or round((time.perf_counter() - started) * 1000, 1)
                call["status"] = exc.reason
                exhausted = len(record["llm_calls"]) >= s.max_model_calls
                if isinstance(exc, OutputValidationError):
                    if regenerations >= s.max_regenerations or exhausted:
                        raise
                    regenerations += 1
                elif exc.retryable:
                    if transient >= s.max_transient_retries or exhausted:
                        raise
                    delay = s.retry_delays_seconds[min(transient, len(s.retry_delays_seconds) - 1)]
                    transient += 1
                    self._sleep(delay)
                else:
                    raise

    def _error_outcome(self, request_id: str, exc: PipelineError, stage: str, record: dict) -> Outcome:
        calls = record["llm_calls"]
        record["error"] = {"reason": exc.reason, "error_stage": stage, "error_message": exc.message[:300],
                           "retry_count": max(0, len(calls) - 1), "is_retryable": exc.retryable}
        response = {"request_id": request_id, "status": "error", "answer": GENERIC_ERROR_MESSAGE,
                    "reason": exc.reason, "citations": []}
        return Outcome(exc.http_status, response, record)
