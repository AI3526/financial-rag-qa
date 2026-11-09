"""STEP 7 API 테스트: FastAPI TestClient + 가짜 검색·생성기 (네트워크·모델 불필요)."""
import json
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from src.api import pipeline as pl
from src.api.app import create_app
from src.api.pipeline import GenerationOutput, PipelineSettings, RagPipeline
from src.api.recording import RunRecorder
from src.api.tracing import parse_traceparent
from src.retrieval.dense import InvalidQuery, RetrievalError, RetrievalResult

TRACE_ID, SPAN_ID = "a" * 32, "b" * 16


def _chunk(i, book="FB01", page=1, text="본문"):
    return {"rank": i, "score": 0.9 - i / 100, "chunk_id": f"{book}-p{page}-c{i}", "book_id": book,
            "file_name": f"{book}.pdf", "page": page, "text": f"{text}{i}", "metadata": {}}


class FakeRetriever:
    def __init__(self, chunks=None, error=None):
        self.chunks = [_chunk(1, "FB01", 2), _chunk(2, "FB02", 5)] if chunks is None else chunks
        self.error, self.calls = error, []

    def retrieve(self, question, k=5):
        self.calls.append((question, k))
        if self.error:
            raise self.error
        return RetrievalResult(question=question, k=k, chunks=self.chunks, durations_ms={"embed": 1.0, "search": 2.0, "total": 3.0})


class ScriptedGenerator:
    """호출마다 steps를 하나씩 소비한다. 예외면 raise, dict면 LLM 출력으로 반환."""

    def __init__(self, *steps):
        self.steps, self.calls = list(steps), []

    def generate(self, question, sources):
        self.calls.append((question, sources))
        step = self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]
        if isinstance(step, Exception):
            raise step
        return GenerationOutput(parsed=step, raw=json.dumps(step, ensure_ascii=False), model_input=[{"role": "user"}],
                                model_id="fake-model", usage={"input_tokens": 100, "output_tokens": 20, "reasoning_tokens": 0})


OK = {"status": "answered", "answer": "답변", "reason": "근거 있음", "source_ids": ["S2"]}


class Span:
    def __init__(self, name, kwargs):
        self.name, self.kwargs, self.updates, self.trace_id = name, kwargs, [], TRACE_ID

    def update(self, **kw):
        self.updates.append(kw)


class FakeLangfuse:
    def __init__(self):
        self.spans = []

    @contextmanager
    def start_as_current_observation(self, **kwargs):
        s = Span(kwargs.get("name"), kwargs)
        self.spans.append(s)
        yield s


class BrokenLangfuse:
    def start_as_current_observation(self, **kwargs):
        raise RuntimeError("langfuse down")


def make(generator=None, retriever=None, langfuse=None, tmp_path=None, **settings):
    sleeps = []
    pipe = RagPipeline(retriever or FakeRetriever(), generator or ScriptedGenerator(OK),
                       PipelineSettings(**settings), sleep=sleeps.append)
    recorder = RunRecorder(tmp_path) if tmp_path else None
    return TestClient(create_app(pipe, langfuse=langfuse, recorder=recorder), raise_server_exceptions=False), sleeps


def post(client, body=None, **headers):
    body = {"question": "영업이익은?", "rag_method": "dense"} if body is None else body
    return client.post("/rag/query", json=body, headers=headers)


def records(tmp_path, name="service"):
    path = tmp_path / name / "runs.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


# --- 정상과 입력 검증 ---
def test_health():
    client, _ = make()
    assert client.get("/health").json() == {"status": "ok"}


def test_normal_question_returns_answer_with_citations_from_code_table():
    client, _ = make()
    r = post(client)
    body = r.json()
    assert r.status_code == 200 and body["status"] == "answered"
    assert body["citations"] == [{"source_id": "S2", "book_id": "FB02", "file_name": "FB02.pdf", "page": 5}]
    assert body["request_id"] == r.headers["X-Request-Id"]
    assert set(body) == {"request_id", "status", "answer", "reason", "citations"}


@pytest.mark.parametrize("body", [
    {"question": "", "rag_method": "dense"}, {"question": "   ", "rag_method": "dense"},
    {"rag_method": "dense"}, {"question": "질문"}, {"question": 123, "rag_method": "dense"},
    {"question": "질문", "rag_method": "hybrid"}, {"question": "질문", "rag_method": "hybrid_rerank"},
    {"question": "질문", "rag_method": "other"},
])
def test_invalid_requests_return_422_in_the_common_error_format(body):
    retriever, generator = FakeRetriever(), ScriptedGenerator(OK)
    client, _ = make(generator, retriever)
    r = post(client, body)
    j = r.json()
    assert r.status_code == 422
    assert j["status"] == "error" and j["reason"] == "invalid_request" and j["citations"] == [] and j["request_id"]
    assert retriever.calls == [] and generator.calls == []  # 검색·생성까지 가지 않는다


@pytest.mark.parametrize("raw", [b'{"question": "\xb0\xa1\xb3\xaa", "rag_method": "dense"}', b"{bad json", b"", b"[1, 2]"])
def test_unreadable_body_returns_422_in_the_common_error_format(raw, tmp_path):
    client, _ = make(tmp_path=tmp_path)
    r = client.post("/rag/query", content=raw, headers={"Content-Type": "application/json"})
    j = r.json()
    assert r.status_code == 422 and j["reason"] == "invalid_request" and j["status"] == "error" and j["request_id"]
    assert "detail" not in j  # FastAPI 기본 오류 형식이 새어 나오지 않는다


def test_unknown_route_keeps_default_404():
    client, _ = make()
    assert client.get("/nope").status_code == 404


def test_invalid_experiment_header_is_rejected_and_not_used_as_path(tmp_path):
    client, _ = make(tmp_path=tmp_path)
    r = post(client, **{"X-Experiment-Id": "../evil", "X-Track": "development"})
    assert r.status_code == 422 and r.json()["reason"] == "invalid_request"
    assert not (tmp_path.parent / "evil").exists() and (tmp_path / "service" / "runs.jsonl").exists()
    assert post(client, **{"X-Track": "weird"}).status_code == 422
    assert post(client, **{"X-Run-Type": "weird"}).status_code == 422


# --- 근거 부족, 검색 ---
def test_insufficient_evidence_is_abstained_not_an_error():
    gen = ScriptedGenerator({"status": "abstained", "answer": "확인할 수 없습니다.", "reason": "자료 없음", "source_ids": []})
    client, _ = make(gen)
    r = post(client)
    assert r.status_code == 200 and r.json()["status"] == "abstained" and r.json()["citations"] == []


def test_empty_retrieval_still_goes_to_generation_with_no_sources():
    gen = ScriptedGenerator({"status": "abstained", "answer": "검색한 자료에서 확인하지 못했습니다.", "reason": "빈 검색", "source_ids": []})
    client, _ = make(gen, FakeRetriever(chunks=[]))
    r = post(client)
    assert r.status_code == 200 and gen.calls[0][1] == []


def test_retrieval_failure_returns_500_retrieval_error():
    client, _ = make(retriever=FakeRetriever(error=RetrievalError("qdrant down")))
    r = post(client)
    assert r.status_code == 500 and r.json()["reason"] == "retrieval_error" and r.json()["status"] == "error"


def test_retriever_invalid_query_maps_to_invalid_request():
    client, _ = make(retriever=FakeRetriever(error=InvalidQuery("비어 있음")))
    assert post(client).status_code == 422


# --- 모델 오류, timeout, 재시도 ---
def test_transient_model_error_is_retried_then_succeeds():
    gen = ScriptedGenerator(pl.ModelError("연결 오류"), pl.ModelError("429"), OK)
    client, sleeps = make(gen)
    r = post(client)
    assert r.status_code == 200 and len(gen.calls) == 3
    assert sleeps == [1.0, 2.0]  # D-030 간격


def test_persistent_model_error_stops_at_max_calls_with_502():
    gen = ScriptedGenerator(pl.ModelError("계속 실패"))
    client, _ = make(gen)
    r = post(client)
    assert r.status_code == 502 and r.json()["reason"] == "model_error" and len(gen.calls) == 3


def test_model_timeout_returns_504_after_retries():
    gen = ScriptedGenerator(pl.ModelTimeout("시간 초과"))
    client, _ = make(gen)
    r = post(client)
    assert r.status_code == 504 and r.json()["reason"] == "model_timeout" and len(gen.calls) == 3


def test_unconfigured_generator_never_invents_an_answer(tmp_path):
    client, _ = make(generator=pl.UnavailableGenerator(), tmp_path=tmp_path)
    r = post(client)
    assert r.status_code == 500 and r.json()["reason"] == "internal_error" and r.json()["answer"] == pl.GENERIC_ERROR_MESSAGE
    assert records(tmp_path)[0]["llm_calls"] == []  # 모델을 호출하지 않았으므로 호출 기록이 없다


# --- 출력 검증 ---
@pytest.mark.parametrize("bad", [
    {"status": "answered", "answer": "a", "reason": "r", "source_ids": ["S9"]},  # 문맥에 없는 ID
    {"status": "answered", "answer": "a", "reason": "r"},  # 필수 필드 누락
    {"status": "error", "answer": "a", "reason": "r", "source_ids": []},  # LLM은 error를 반환할 수 없다
    {"status": "answered", "answer": "a", "reason": "r", "source_ids": "S1"},  # 자료형
    ["not", "an", "object"],
])
def test_invalid_llm_output_regenerates_once_then_returns_502(bad):
    gen = ScriptedGenerator(bad)
    client, _ = make(gen)
    r = post(client)
    assert r.status_code == 502 and r.json()["reason"] == "output_validation_error"
    assert len(gen.calls) == 2  # 처음 + 재생성 1회


def test_regeneration_can_recover():
    gen = ScriptedGenerator({"status": "answered", "answer": "a", "reason": "r", "source_ids": ["S9"]}, OK)
    client, _ = make(gen)
    r = post(client)
    assert r.status_code == 200 and len(gen.calls) == 2


def test_total_model_calls_never_exceed_max_calls():
    gen = ScriptedGenerator(pl.ModelError("x"), {"status": "answered", "answer": "a", "reason": "r", "source_ids": ["S9"]})
    client, _ = make(gen, max_model_calls=3)
    post(client)
    assert len(gen.calls) <= 3


# --- 예상 밖 오류, 비밀 ---
def test_unexpected_exception_returns_500_without_leaking_details():
    class Exploding:
        def generate(self, question, sources):
            raise RuntimeError("sk-SECRET-should-not-leak /internal/path")

    client, _ = make(Exploding())
    r = post(client)
    assert r.status_code == 500 and r.json()["reason"] == "internal_error"
    assert "SECRET" not in r.text and "internal/path" not in r.text


# --- 평가 데이터 유입 방지 ---
def test_extra_evaluation_fields_in_request_never_reach_retrieval_or_generation():
    retriever, gen = FakeRetriever(), ScriptedGenerator(OK)
    client, _ = make(gen, retriever)
    r = post(client, {"question": "질문", "rag_method": "dense", "question_id": "F-DEV-01",
                      "reference_answer": "정답!", "expected_action": "answer", "evidence_groups": [[1]]})
    assert r.status_code == 200
    assert retriever.calls == [("질문", 5)]
    assert "정답!" not in json.dumps(gen.calls, ensure_ascii=False, default=str) and gen.calls[0][0] == "질문"


# --- run 레코드 ---
def test_run_record_is_saved_with_required_fields_and_llm_calls(tmp_path):
    gen = ScriptedGenerator(pl.ModelError("일시 오류"), OK)
    client, _ = make(gen, tmp_path=tmp_path)
    r = post(client, {"question": "영업이익은?", "rag_method": "dense", "question_id": "Q1"},
             **{"X-Experiment-Id": "exp-001", "X-Track": "development", "X-Run-Type": "general_eval"})
    rec = records(tmp_path, "exp-001")[0]
    assert rec["request_id"] == r.json()["request_id"] and rec["experiment_id"] == "exp-001"
    assert rec["question_id"] == "Q1" and rec["track"] == "development" and rec["run_type"] == "general_eval"
    assert rec["http_status"] == 200 and rec["response"]["status"] == "answered"
    assert [c["status"] for c in rec["llm_calls"]] == ["model_error", "ok"]  # 시도마다 1행
    assert rec["llm_calls"][1]["input_tokens"] == 100 and rec["llm_calls"][0]["input_tokens"] is None  # 모르면 null
    assert rec["retrieval"]["dense"][0]["chunk_id"] == "FB01-p2-c1" and rec["retrieval"]["sparse"] is None
    assert rec["final_context"]["chunks"][1]["source_id"] == "S2"
    assert rec["performance"]["retry_count"] == 1 and rec["performance"]["final_status"] == "answered"
    assert rec["model_input"] is not None and rec["llm_output_parsed"]["source_ids"] == ["S2"]
    assert "reference_answer" not in rec and "expected_action" not in rec


def test_failed_and_invalid_requests_are_also_recorded(tmp_path):
    client, _ = make(retriever=FakeRetriever(error=RetrievalError("down")), tmp_path=tmp_path)
    post(client)
    post(client, {"question": "", "rag_method": "dense"})
    post(client, {"rag_method": "dense"})
    recs = records(tmp_path)
    assert [r["http_status"] for r in recs] == [500, 422, 422]
    assert recs[0]["error"]["reason"] == "retrieval_error" and recs[0]["error"]["error_stage"] == "retrieval"
    assert recs[1]["error"]["reason"] == "invalid_request"


def test_records_are_appended_not_overwritten(tmp_path):
    client, _ = make(tmp_path=tmp_path)
    for _ in range(3):
        post(client)
    assert len(records(tmp_path)) == 3


# --- Langfuse ---
def test_traceparent_continues_the_given_trace_and_records_ids(tmp_path):
    lf = FakeLangfuse()
    client, _ = make(langfuse=lf, tmp_path=tmp_path)
    r = post(client, traceparent=f"00-{TRACE_ID}-{SPAN_ID}-01")
    assert r.status_code == 200 and r.headers["X-Trace-Id"] == TRACE_ID
    span = lf.spans[0]
    assert span.name == "rag-query" and span.kwargs["trace_context"] == {"trace_id": TRACE_ID, "parent_span_id": SPAN_ID}
    assert span.updates[0]["output"]["status"] == "answered"
    rec = records(tmp_path)[0]
    assert rec["trace_id"] == TRACE_ID and rec["observability"] == {"langfuse_recorded": True, "langfuse_error": None}


def test_without_traceparent_a_new_trace_is_created():
    lf = FakeLangfuse()
    client, _ = make(langfuse=lf)
    r = post(client)
    assert "trace_context" not in lf.spans[0].kwargs and r.headers["X-Trace-Id"] == TRACE_ID


def test_malformed_traceparent_is_ignored():
    for bad in ("garbage", f"00-{'0' * 32}-{SPAN_ID}-01", f"00-{TRACE_ID}-{'0' * 16}-01", f"01-{TRACE_ID}-{SPAN_ID}-01", ""):
        assert parse_traceparent(bad) is None
    assert parse_traceparent(f"00-{TRACE_ID}-{SPAN_ID}-00") == {"trace_id": TRACE_ID, "parent_span_id": SPAN_ID}
    lf = FakeLangfuse()
    client, _ = make(langfuse=lf)
    post(client, traceparent="garbage")
    assert "trace_context" not in lf.spans[0].kwargs


def test_error_response_marks_trace_as_error():
    lf = FakeLangfuse()
    client, _ = make(retriever=FakeRetriever(error=RetrievalError("down")), langfuse=lf)
    post(client)
    assert lf.spans[0].updates[0]["level"] == "ERROR" and lf.spans[0].updates[0]["status_message"] == "retrieval_error"


def test_langfuse_failure_does_not_change_answer_and_record_is_saved(tmp_path):
    client, _ = make(langfuse=BrokenLangfuse(), tmp_path=tmp_path)
    r = post(client)
    assert r.status_code == 200 and r.json()["status"] == "answered"
    rec = records(tmp_path)[0]
    assert rec["observability"]["langfuse_recorded"] is False and rec["observability"]["langfuse_error"] == "RuntimeError"


def test_recorder_failure_does_not_break_the_response(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("파일이 폴더 자리를 막고 있음", encoding="utf-8")
    client, _ = make(tmp_path=blocker)
    assert post(client).status_code == 200
