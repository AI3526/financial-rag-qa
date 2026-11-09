"""gpt-6-luna 생성기 (STEP 8, 08 §12). `src.api.pipeline.Generator` 인터페이스를 구현한다.

- LangChain `ChatPromptTemplate` + `ChatOpenAI`. 재시도는 파이프라인이 담당하므로 `max_retries=0`이다(중복 재시도 금지).
- 출력은 JSON schema(strict) 하나: `{status, answer, reason, source_ids}`. 검증은 파이프라인의 `validate_output`이 한다.
- 오류는 파이프라인이 아는 오류로 바꾼다. 오류 메시지에는 원본 예외 문구를 넣지 않는다(키 일부가 섞일 수 있음).
- 평가 정답·근거는 이 모듈에 들어오지 않는다. 입력은 질문과 검색 문맥뿐이다.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable

import openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.api.pipeline import (GenerationOutput, ModelError, ModelTimeout, OutputValidationError, PipelineError,
                              PipelineSettings)
from src.generation.context import format_context
from src.generation.prompts import PromptSpec, load_prompt

RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "rag_answer", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["status", "answer", "reason", "source_ids"],
    "properties": {
        "status": {"type": "string", "enum": ["answered", "needs_clarification", "abstained"]},
        "answer": {"type": "string"}, "reason": {"type": "string"},
        "source_ids": {"type": "array", "items": {"type": "string"}}}}}}

_TRUNCATION_MARKER = "output limit was reached"  # 잘림은 finish_reason이 아니라 400 오류 문구로만 구분된다(STEP 8 확인)


def map_model_error(exc: Exception) -> PipelineError:
    """OpenAI 호출 예외를 파이프라인 오류로 바꾼다. 재시도 가능 여부(retryable)도 정한다."""
    name = type(exc).__name__
    if isinstance(exc, openai.APITimeoutError):  # APIConnectionError의 하위 클래스라 먼저 검사한다
        return ModelTimeout(f"{name}")
    if isinstance(exc, openai.APIConnectionError):
        return ModelError(f"{name}", retryable=True)
    if isinstance(exc, openai.RateLimitError):
        return ModelError(f"{name} status=429", retryable=True)
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        if status >= 500:
            return ModelError(f"{name} status={status}", retryable=True)
        if isinstance(exc, openai.BadRequestError) and _TRUNCATION_MARKER in str(exc):
            return OutputValidationError("출력이 max_completion_tokens에서 잘렸습니다.")
        return ModelError(f"{name} status={status}", retryable=False)  # 400, 401, 403, 404 등은 재시도해도 같다
    return ModelError(f"{name}", retryable=False)


def build_llm(settings: PipelineSettings, model: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or os.environ["OPENAI_MODEL"], max_retries=0, timeout=settings.model_timeout_seconds,
        reasoning_effort=settings.reasoning_effort, max_completion_tokens=settings.max_completion_tokens,
        seed=settings.seed, model_kwargs={"response_format": RESPONSE_FORMAT},
    )


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content or [])


def _usage(ai_message) -> dict | None:
    u = getattr(ai_message, "usage_metadata", None)
    if not u:
        return None
    details = u.get("input_token_details") or {}
    return {"input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"),
            "reasoning_tokens": (u.get("output_token_details") or {}).get("reasoning"),
            # 입력 중 캐시에서 읽은 토큰(같은 접두어를 반복하면 생김)과 캐시에 쓴 토큰. 시간·비용 계산에 필요하다
            "cached_input_tokens": details.get("cache_read"), "cache_write_tokens": details.get("cache_creation"),
            "service_tier": (getattr(ai_message, "response_metadata", None) or {}).get("service_tier")}


class LangChainGenerator:
    def __init__(self, settings: PipelineSettings | None = None, prompt: PromptSpec | None = None, *,
                 model: str | None = None, llm=None, callbacks_factory: Callable[[], list] | None = None):
        self.settings = settings or PipelineSettings()
        self.prompt = prompt or load_prompt()
        self.template = ChatPromptTemplate.from_messages([("system", self.prompt.system), ("user", self.prompt.user)])
        self.llm = llm or build_llm(self.settings, model)
        self.callbacks_factory = callbacks_factory

    @property
    def prompt_version(self) -> str:
        return self.prompt.version

    @property
    def prompt_sha256(self) -> str:
        return self.prompt.sha256

    def generate(self, question: str, sources: list[dict]) -> GenerationOutput:
        messages = self.template.format_messages(context=format_context(sources), question=question)
        config = {"callbacks": self.callbacks_factory()} if self.callbacks_factory else None
        try:
            ai = self.llm.invoke(messages, config=config)
        except PipelineError:
            raise
        except Exception as exc:  # noqa: BLE001 - 종류를 파이프라인 오류로 통일한다
            raise map_model_error(exc) from exc
        meta = getattr(ai, "response_metadata", None) or {}
        if meta.get("finish_reason") == "length":
            raise OutputValidationError("출력이 max_completion_tokens에서 잘렸습니다.")
        raw = _text(ai.content)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None  # validate_output이 "JSON 객체가 아님"으로 처리한다
        return GenerationOutput(
            parsed=parsed, raw=raw,
            model_input=[{"role": {"human": "user"}.get(m.type, m.type), "content": m.content} for m in messages],
            model_id=meta.get("model_name"), usage=_usage(ai))
