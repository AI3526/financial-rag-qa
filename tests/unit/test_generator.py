import json

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage

from src.api import pipeline as pl
from src.api.pipeline import PipelineSettings, RagPipeline
from src.generation.generator import LangChainGenerator, map_model_error
from src.generation.pricing import load_pricing
from tests.unit.test_api import FakeRetriever

REQ = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def status_error(cls, status, message="error"):
    resp = httpx.Response(status, request=REQ, json={"error": {"message": message}})
    return cls(message, response=resp, body={"message": message})


OK = {"status": "answered", "answer": "96억원입니다.", "reason": "S2 근거", "source_ids": ["S2"]}


def ai(payload=OK, finish="stop"):
    content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return AIMessage(content=content,
                     usage_metadata={"input_tokens": 120, "output_tokens": 40, "total_tokens": 160,
                                     "input_token_details": {"cache_read": 64},
                                     "output_token_details": {"reasoning": 7}},
                     response_metadata={"finish_reason": finish, "model_name": "gpt-6-luna"})


class FakeLLM:
    """steps를 호출마다 하나씩 소비한다(마지막 값은 반복). 예외면 raise."""

    def __init__(self, *steps):
        self.steps, self.calls = list(steps), []

    def invoke(self, messages, config=None):
        self.calls.append((messages, config))
        step = self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]
        if isinstance(step, Exception):
            raise step
        return step


def run(llm, **settings):
    pipe = RagPipeline(FakeRetriever(), LangChainGenerator(PipelineSettings(**settings), llm=llm), PipelineSettings(**settings),
                       sleep=lambda s: None)
    return pipe.run("rid", "영업이익은?", "dense")


# --- 정상, 사용량, 프롬프트 ---
def test_success_records_usage_model_input_and_reasoning_tokens():
    llm = FakeLLM(ai())
    out = run(llm)
    assert out.http_status == 200 and out.response["citations"][0]["file_name"] == "FB02.pdf"
    call = out.record["llm_calls"][0]
    assert (call["input_tokens"], call["output_tokens"], call["reasoning_tokens"]) == (120, 40, 7)
    assert call["cached_input_tokens"] == 64  # 입력 120 중 캐시에서 읽은 토큰
    assert call["model_id"] == "gpt-6-luna" and call["status"] == "ok"
    roles = [m["role"] for m in out.record["model_input"]]
    assert roles == ["system", "user"]
    user = out.record["model_input"][1]["content"]
    assert "[S1] 본문1" in user and "[S2] 본문2" in user and "영업이익은?" in user
    assert out.record["llm_output_raw"] and out.record["final_context"]["context_limit_tokens"] == 4000


def test_cache_write_service_tier_and_cost_are_recorded():
    msg = AIMessage(content=json.dumps(OK), usage_metadata={
        "input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050,
        "input_token_details": {"cache_read": 600, "cache_creation": 100}, "output_token_details": {"reasoning": 0}},
        response_metadata={"finish_reason": "stop", "model_name": "gpt-6-luna", "service_tier": "default"})
    pipe = RagPipeline(FakeRetriever(), LangChainGenerator(llm=FakeLLM(msg)), PipelineSettings(),
                       sleep=lambda s: None, pricing=load_pricing())
    rec = pipe.run("rid", "영업이익은?", "dense").record
    call = rec["llm_calls"][0]
    assert (call["cached_input_tokens"], call["cache_write_tokens"], call["service_tier"]) == (600, 100, "default")
    assert rec["cost_status"] == "complete" and rec["currency"] == "USD"
    assert abs(rec["cost"] - (300 * 0.10 + 600 * 0.01 + 100 * 0.125 + 50 * 0.50) / 1e6) < 1e-12


def test_cost_is_null_without_pricing_and_when_usage_is_missing():
    assert run(FakeLLM(ai())).record["cost"] is None  # 단가를 주지 않으면 계산하지 않는다
    msg = AIMessage(content=json.dumps(OK), response_metadata={"finish_reason": "stop", "model_name": "gpt-6-luna"})
    pipe = RagPipeline(FakeRetriever(), LangChainGenerator(llm=FakeLLM(msg)), PipelineSettings(),
                       sleep=lambda s: None, pricing=load_pricing())
    rec = pipe.run("rid", "영업이익은?", "dense").record
    assert rec["cost"] is None and rec["cost_status"] == "unknown"  # 사용량을 모르면 0이 아니라 null


def test_missing_usage_is_null_not_zero():
    msg = AIMessage(content=json.dumps(OK), response_metadata={"finish_reason": "stop", "model_name": "gpt-6-luna"})
    call = run(FakeLLM(msg)).record["llm_calls"][0]
    assert call["input_tokens"] is None and call["output_tokens"] is None and call["usage_status"] == "unknown"
    assert call["cached_input_tokens"] is None


def test_question_with_braces_does_not_break_the_prompt_template():
    llm = FakeLLM(ai())
    pipe = RagPipeline(FakeRetriever(), LangChainGenerator(llm=llm), PipelineSettings(), sleep=lambda s: None)
    out = pipe.run("rid", "{context} {question} {unknown} 는?", "dense")
    assert out.http_status == 200 and "{unknown}" in llm.calls[0][0][1].content


def test_prompt_contains_only_question_context_and_instructions():
    llm = FakeLLM(ai())
    LangChainGenerator(llm=llm).generate("질문?", [{"source_id": "S1", "text": "자료"}])
    text = " ".join(m.content for m in llm.calls[0][0])
    for forbidden in ("reference_answer", "expected_action", "evidence_groups", "required_facts", "forbidden_claims"):
        assert forbidden not in text


def test_langfuse_callbacks_are_passed_when_a_factory_is_given():
    llm = FakeLLM(ai())
    marker = object()
    LangChainGenerator(llm=llm, callbacks_factory=lambda: [marker]).generate("q", [])
    assert llm.calls[0][1] == {"callbacks": [marker], "run_name": "answer-generation"}  # Langfuse 관측 이름(05 §3.4)


# --- 오류 매핑과 재시도 ---
@pytest.mark.parametrize("exc, cls, retryable", [
    (openai.APITimeoutError(request=REQ), pl.ModelTimeout, True),
    (openai.APIConnectionError(request=REQ), pl.ModelError, True),
    (status_error(openai.RateLimitError, 429), pl.ModelError, True),
    (status_error(openai.InternalServerError, 500), pl.ModelError, True),
    (status_error(openai.InternalServerError, 503), pl.ModelError, True),
    (status_error(openai.BadRequestError, 400, "Unsupported value"), pl.ModelError, False),
    (status_error(openai.AuthenticationError, 401, "Incorrect API key sk-SECRET123"), pl.ModelError, False),
    (status_error(openai.PermissionDeniedError, 403), pl.ModelError, False),
    (status_error(openai.NotFoundError, 404), pl.ModelError, False),
    (RuntimeError("알 수 없는 오류 sk-SECRET123"), pl.ModelError, False),
])
def test_error_mapping_and_retryability(exc, cls, retryable):
    mapped = map_model_error(exc)
    assert type(mapped) is cls and mapped.retryable is retryable
    assert "SECRET" not in mapped.message  # 원본 예외 문구(키 일부 포함 가능)를 옮기지 않는다


def test_truncated_output_is_an_output_error_not_a_model_error():
    err = status_error(openai.BadRequestError, 400,
                       "Could not finish the message because max_tokens or model output limit was reached.")
    assert type(map_model_error(err)) is pl.OutputValidationError


def test_truncation_via_finish_reason_length_is_not_passed_as_an_answer():
    out = run(FakeLLM(ai(finish="length")))
    assert out.http_status == 502 and out.response["reason"] == "output_validation_error"
    assert len(out.record["llm_calls"]) == 2  # 처음 + 재생성 1회


def test_non_retryable_error_is_called_once_and_returns_502():
    llm = FakeLLM(status_error(openai.AuthenticationError, 401))
    out = run(llm)
    assert out.http_status == 502 and out.response["reason"] == "model_error" and len(llm.calls) == 1


def test_transient_errors_are_retried_until_success():
    llm = FakeLLM(status_error(openai.RateLimitError, 429), openai.APIConnectionError(request=REQ), ai())
    out = run(llm)
    assert out.http_status == 200 and len(llm.calls) == 3
    assert [c["status"] for c in out.record["llm_calls"]] == ["model_error", "model_error", "ok"]


def test_timeouts_stop_at_the_call_limit_with_504():
    llm = FakeLLM(openai.APITimeoutError(request=REQ))
    out = run(llm)
    assert out.http_status == 504 and out.response["reason"] == "model_timeout" and len(llm.calls) == 3


def test_non_json_content_is_an_output_error():
    out = run(FakeLLM(ai("JSON이 아닙니다")))
    assert out.http_status == 502 and out.response["reason"] == "output_validation_error"


def test_source_id_outside_the_final_context_is_rejected():
    out = run(FakeLLM(ai({**OK, "source_ids": ["S9"]})))
    assert out.http_status == 502 and out.response["reason"] == "output_validation_error"


def test_error_record_never_contains_secret_text():
    out = run(FakeLLM(status_error(openai.AuthenticationError, 401, "Incorrect API key sk-SECRET123")))
    assert "SECRET" not in json.dumps(out.record, ensure_ascii=False) + json.dumps(out.response, ensure_ascii=False)
