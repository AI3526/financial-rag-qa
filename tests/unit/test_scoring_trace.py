from contextlib import contextmanager

import pytest

from src.evaluation.scoring_trace import record_scoring
from tests.unit.test_api import BrokenLangfuse, FakeLangfuse


@pytest.fixture()
def propagated(monkeypatch):
    import langfuse

    calls = []

    @contextmanager
    def fake(**kwargs):
        calls.append(kwargs)
        yield

    monkeypatch.setattr(langfuse, "propagate_attributes", fake)
    return calls


KW = dict(name="faithfulness-evaluation", model="gpt-6-luna", track="development", rag_method="hybrid",
          evaluated={"request_id": "req-1", "trace_id": "t" * 32, "question_id": "F-DEV-01"},
          input={"question": "q"}, output={"status": "scored", "score": 1.0},
          usage={"input_tokens": 2700, "output_tokens": 750}, metadata={"evaluator": "ragas-faithfulness", "prompt_version": "v1"})


def test_scoring_is_a_separate_generation_tagged_run_type_scoring_and_linked_to_the_evaluated_request(propagated):
    lf = FakeLangfuse()
    record_scoring(lf, **KW)
    attrs = propagated[0]
    assert attrs["trace_name"] == "faithfulness-evaluation" and attrs["environment"] == "development"
    assert "run_type:scoring" in attrs["tags"] and "rag_method:hybrid" in attrs["tags"] and "track:development" in attrs["tags"]
    md = attrs["metadata"]
    assert md["run_type"] == "scoring" and md["evaluated_request_id"] == "req-1" and md["evaluated_trace_id"] == "t" * 32
    span = lf.spans[0]
    assert span.name == "faithfulness-evaluation" and span.kwargs["as_type"] == "generation" and span.kwargs["model"] == "gpt-6-luna"
    assert span.kwargs["usage_details"] == {"input": 2700, "output": 750}


def test_scoring_trace_is_never_the_service_root_or_answer_generation(propagated):
    lf = FakeLangfuse()
    record_scoring(lf, **KW)
    names = [s.name for s in lf.spans]
    assert "rag-query" not in names and "answer-generation" not in names and "trace_context" not in lf.spans[0].kwargs


def test_missing_usage_is_not_recorded_as_zero(propagated):
    lf = FakeLangfuse()
    record_scoring(lf, **{**KW, "usage": None})
    assert lf.spans[0].kwargs["usage_details"] is None


def test_no_langfuse_or_broken_langfuse_returns_none_without_raising(propagated):
    assert record_scoring(None, **KW) is None
    assert record_scoring(BrokenLangfuse(), **KW) is None
