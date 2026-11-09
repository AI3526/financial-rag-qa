"""STEP 10: Langfuse에 남는 관측 기록 검증(가짜 Langfuse). 05_OBSERVABILITY_PLAN.md의 필수 항목."""
import json
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from src.api import pipeline as pl
from src.api.app import create_app
from src.api.pipeline import GenerationOutput, PipelineSettings, RagPipeline
from src.retrieval.dense import RetrievalError
from tests.unit.test_api import FakeLangfuse, FakeRetriever, ScriptedGenerator, OK, _chunk


class NamedFake(FakeRetriever):
    def describe(self):
        return {"final_k": 5}


def make(generator=None, retriever=None, settings=None, lf=None):
    lf = lf or FakeLangfuse()
    pipe = RagPipeline(retriever or NamedFake(), generator or ScriptedGenerator(OK), settings or PipelineSettings(),
                       sleep=lambda s: None, langfuse=lf)
    return TestClient(create_app(pipe, langfuse=lf), raise_server_exceptions=False), lf, pipe


def post(client, body=None, **headers):
    return client.post("/rag/query", json=body or {"question": "영업이익은?", "rag_method": "dense"}, headers=headers)


def span(lf, name):
    return next(s for s in lf.spans if s.name == name)


@pytest.fixture()
def propagated(monkeypatch):
    """Trace 전체에 전파되는 속성(metadata·tags·environment)은 `propagate_attributes`로 기록된다. 그 호출을 잡아 둔다."""
    import langfuse

    calls = []

    @contextmanager
    def fake(**kwargs):
        calls.append(kwargs)
        yield

    monkeypatch.setattr(langfuse, "propagate_attributes", fake)
    return calls


def test_successful_request_records_the_required_steps_under_one_root():
    client, lf, _ = make()
    assert post(client).status_code == 200
    names = [s.name for s in lf.spans]
    assert names == ["rag-query", "context-construction", "citation-and-status"]  # 한 요청에 루트 하나, 단계는 하위 Span
    assert names.count("rag-query") == 1


def test_context_construction_separates_retrieved_chunks_from_the_final_context():
    chunks = [_chunk(i, "FB01", i, "가" * 30) for i in range(1, 4)]
    client, lf, _ = make(retriever=NamedFake(chunks), settings=PipelineSettings(context_limit_tokens=70))
    post(client)
    cc = span(lf, "context-construction")
    assert [r["chunk_id"] for r in cc.kwargs["input"]["retrieved"]] == [c["chunk_id"] for c in chunks]  # 검색 결과 전체
    out = cc.updates[0]["output"]
    assert out["retrieved_count"] == 3 and out["final_count"] < 3  # 한도 때문에 일부만 모델에 전달됨
    assert out["dropped_chunk_ids"] and set(out["dropped_chunk_ids"]).isdisjoint(c["chunk_id"] for c in out["final_chunks"])
    assert [c["source_id"] for c in out["final_chunks"]] == [f"S{i}" for i in range(1, out["final_count"] + 1)]
    assert out["context_limit_tokens"] == 70 and out["token_count_method"].endswith("(estimate)")
    assert all({"file_name", "page", "book_id"} <= set(c) for c in out["final_chunks"])


def test_citation_and_status_span_records_validation_and_source_mapping():
    client, lf, _ = make()
    post(client)
    cs = span(lf, "citation-and-status")
    assert cs.kwargs["input"]["llm_source_ids"] == ["S2"] and cs.kwargs["input"]["context_source_ids"] == ["S1", "S2"]
    out = cs.updates[0]["output"]
    assert out["validation"] == "passed" and out["status"] == "answered"
    assert out["citations"] == [{"source_id": "S2", "book_id": "FB02", "file_name": "FB02.pdf", "page": 5}]


def test_root_trace_carries_identifiers_versions_and_tags(propagated):
    client, lf, pipe = make()
    r = post(client, **{"X-Experiment-Id": "exp-1", "X-Track": "development", "X-Run-Type": "general_eval"})
    attrs = propagated[0]
    md = attrs["metadata"]
    assert attrs["trace_name"] == "rag-query" and attrs["environment"] == "development"
    assert set(attrs["tags"]) == {"run_type:general_eval", "track:development", "rag_method:dense"}
    assert md["request_id"] == r.json()["request_id"] and md["experiment_id"] == "exp-1" and md["rag_method"] == "dense"
    assert md["track"] == "development" and md["run_type"] == "general_eval"
    info = pipe.version_info("dense")
    assert md["config_version"] == info["config_version"] and md["config_version"].startswith("cfg-")
    for key in ("dataset_id", "dataset_version", "prompt_version", "prompt_sha256", "embedding_version", "chunking_version"):
        assert (key in md) == (info[key] is not None)  # 값이 있는 식별 정보는 모두 metadata에 있다(가짜 검색기는 값이 없다)


def test_trace_continuing_an_experiment_does_not_override_environment_or_name(propagated):
    client, lf, _ = make()
    post(client, traceparent="00-" + "a" * 32 + "-" + "b" * 16 + "-01")
    attrs = propagated[0]
    assert "environment" not in attrs and "trace_name" not in attrs  # 실험 Trace의 환경·이름을 바꾸지 않는다(D-029)
    assert attrs["metadata"]["track"] == "development" and any(t.startswith("track:") for t in attrs["tags"])


def test_root_output_has_performance_and_cost_fields():
    client, lf, _ = make()
    post(client)
    out = span(lf, "rag-query").updates[0]["output"]
    assert out["status"] == "answered" and out["performance"]["total_duration_ms"] >= 0 and "cost" in out["cost"]
    assert out["performance"]["final_status"] == "answered"


@pytest.mark.parametrize("make_kwargs, stage, reason", [
    ({"retriever": NamedFake(error=RetrievalError("down"))}, "retrieval", "retrieval_error"),
    ({"generator": ScriptedGenerator(pl.ModelTimeout("시간 초과"))}, "generation", "model_timeout"),
    ({"generator": ScriptedGenerator({"status": "answered", "answer": "a", "reason": "r", "source_ids": ["S9"]})}, "generation", "output_validation_error"),
])
def test_error_trace_records_type_stage_and_retry_info(make_kwargs, stage, reason):
    client, lf, _ = make(**make_kwargs)
    post(client)
    root = span(lf, "rag-query")
    upd = root.updates[0]
    assert upd["level"] == "ERROR" and upd["status_message"] == f"{reason} (stage={stage})"
    err = upd["output"]["error"]
    assert err["reason"] == reason and err["error_stage"] == stage and "retry_count" in err and "is_retryable" in err
    assert upd["output"]["status"] == "error"


def test_blank_question_error_is_traced_at_the_validation_stage():
    client, lf, _ = make()
    post(client, {"question": "   ", "rag_method": "dense"})
    upd = span(lf, "rag-query").updates[0]
    assert upd["output"]["error"]["error_stage"] == "validate" and upd["output"]["error"]["reason"] == "invalid_request"


def test_request_validation_errors_are_traced_too(propagated):
    client, lf, _ = make()
    r = client.post("/rag/query", json={"question": "질문"})  # rag_method 누락
    assert r.status_code == 422
    root = span(lf, "rag-query")
    assert root.updates[0]["level"] == "ERROR" and root.updates[0]["output"]["error"]["reason"] == "invalid_request"
    assert propagated[0]["metadata"]["request_id"] == r.json()["request_id"]


def test_config_version_is_stable_and_changes_with_settings_prompt_or_method():
    _, _, a = make()
    _, _, b = make()
    assert a.version_info("dense")["config_version"] == b.version_info("dense")["config_version"]  # 같은 설정이면 같은 값
    _, _, c = make(settings=PipelineSettings(context_limit_tokens=3000))
    assert c.version_info("dense")["config_version"] != a.version_info("dense")["config_version"]
    _, _, d = make(settings=PipelineSettings(reasoning_effort="high"))
    assert d.version_info("dense")["config_version"] != a.version_info("dense")["config_version"]


def test_langfuse_off_or_broken_changes_nothing_in_the_response():
    from tests.unit.test_api import BrokenLangfuse

    plain = TestClient(create_app(RagPipeline(NamedFake(), ScriptedGenerator(OK), PipelineSettings())))
    broken, _, _ = make(lf=BrokenLangfuse())
    assert post(plain).json()["citations"] == post(broken).json()["citations"]


def test_malformed_request_trace_uses_the_run_type_from_headers(propagated):
    client, lf, _ = make()
    client.post("/rag/query", json={"question": "질문"}, headers={"X-Run-Type": "mock_error", "X-Experiment-Id": "e1"})
    attrs = propagated[0]
    assert "run_type:mock_error" in attrs["tags"] and attrs["metadata"]["run_type"] == "mock_error"
    assert attrs["metadata"]["experiment_id"] == "e1"
