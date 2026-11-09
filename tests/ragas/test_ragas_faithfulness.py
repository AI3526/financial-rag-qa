"""ragas가 설치된 격리 환경에서만 실행한다(메인 환경에서는 건너뛴다).

    uv run --with ragas --with "langchain-community<0.4" pytest tests/ragas -q

평가용 LLM은 가짜(OpenAI 호출 없음)로, ragas의 `Faithfulness` 상속 클래스가 승인된 프롬프트를 쓰고 판정 이유를 보존하며
N/A와 평가 불가를 올바르게 처리하는지 확인한다.
"""
import asyncio

import pytest

pytest.importorskip("ragas.metrics.collections")

from ragas.llms.base import InstructorBaseRagasLLM  # noqa: E402
from ragas.metrics.collections.faithfulness.util import NLIStatementOutput, StatementFaithfulnessAnswer, StatementGeneratorOutput  # noqa: E402

from src.evaluation import faithfulness as fa  # noqa: E402
from src.evaluation.ragas_faithfulness import DetailedFaithfulness, evaluate_many, make_statement_prompt  # noqa: E402


class FakeLLM(InstructorBaseRagasLLM):
    """1단계에는 statements를, 2단계에는 verdicts를 돌려주는 가짜. 받은 프롬프트 문자열을 기록한다."""

    def __init__(self, statements, verdicts, fail_at=None):
        self.statements, self.verdicts, self.fail_at = statements, verdicts, fail_at
        self.prompts = []

    async def agenerate(self, prompt, response_model):
        self.prompts.append(prompt)
        step = 1 if response_model is StatementGeneratorOutput else 2
        if self.fail_at == step:
            raise TimeoutError("가짜 시간 초과")
        if step == 1:
            return StatementGeneratorOutput(statements=self.statements)
        return NLIStatementOutput(statements=[StatementFaithfulnessAnswer(statement=s, reason=r, verdict=vd) for s, r, vd in self.verdicts])

    def generate(self, prompt, response_model):
        return asyncio.run(self.agenerate(prompt, response_model))


def run(llm, q="질문?", a="답변.", ctx=("문맥",)):
    metric = DetailedFaithfulness(llm)
    return asyncio.run(metric.evaluate(q, a, list(ctx))), metric


def test_uses_the_approved_statement_prompt_and_the_default_nli_prompt():
    llm = FakeLLM(["x."], [("x.", "이유", 1)])
    _, metric = run(llm)
    step1 = llm.prompts[0]
    assert "(c) say that some information is not available" in step1 and "return an empty list of statements" in step1
    assert "Albert Einstein" in step1 and "알려 주시겠어요" in step1  # 라이브러리 예시와 추가 예시가 모두 들어간다
    assert "can be directly inferred based on the context" in llm.prompts[1]  # 2단계는 ragas 기본 프롬프트
    assert metric.prompt_spec["version"] == "faithfulness-statement-v1"


def test_score_and_per_statement_reasons_are_preserved_and_match_ragas_own_score():
    llm = FakeLLM(["a.", "b.", "c."], [("a.", "근거 있음", 1), ("b.", "근거 있음", 1), ("c.", "문맥에 없음", 0)])
    d, _ = run(llm)
    assert d["status"] == "scored" and abs(d["score"] - 2 / 3) < 1e-9 and abs(d["ragas_score"] - d["score"]) < 1e-9
    assert [s["reason"] for s in d["statements"]] == ["근거 있음", "근거 있음", "문맥에 없음"]  # ragas가 버리던 이유를 보존
    assert d["statements_generated"] == 3 and d["usage"]["calls"] == 0  # 가짜는 연결이 없으므로 사용량 0(실제 실행에서 센다)


def test_no_statements_gives_na_and_skips_the_second_llm_call():
    llm = FakeLLM([], [])
    d, _ = run(llm, a="어느 회사의 어느 기간 실적을 말씀하시는지 알려 주시겠어요?")
    assert d["status"] == "not_applicable" and d["score"] is None and len(llm.prompts) == 1  # 판정 호출 없이 N/A


@pytest.mark.parametrize("step", [1, 2])
def test_llm_failure_is_unevaluable_not_a_score_and_never_raises(step):
    d, _ = run(FakeLLM(["a."], [("a.", "r", 1)], fail_at=step))
    assert d["status"] == "unevaluable" and d["score"] is None and "TimeoutError" in d["reason"]


def test_make_statement_prompt_uses_the_file_contents():
    spec = fa.load_statement_prompt()
    p = make_statement_prompt(spec)
    assert p.instruction == spec["instruction"] and len(p.examples) == 2 and p.examples[1][1].statements == []


def test_evaluate_many_runs_each_answer_for_each_pass_and_keeps_order():
    llm = FakeLLM(["a."], [("a.", "r", 1)])
    metric = DetailedFaithfulness(llm)
    items = [{"request_id": f"r{i}", "question_id": f"Q{i}", "rag_method": "dense", "question": "q", "answer": "a", "contexts": ["c"]} for i in range(3)]
    rows = asyncio.run(evaluate_many(metric, items, passes=3, concurrency=2))
    assert [(r["request_id"], r["pass"]) for r in rows] == [(f"r{i}", n) for i in range(3) for n in (1, 2, 3)]
    assert all(r["status"] == "scored" and r["score"] == 1.0 for r in rows)


def test_ragas_usage_telemetry_is_disabled_and_nothing_is_posted(monkeypatch):
    import ragas._analytics as analytics

    assert analytics.do_not_track() is True  # src.evaluation.ragas_faithfulness를 import하면 꺼진다
    posted = []
    monkeypatch.setattr(analytics.requests, "post", lambda *a, **k: posted.append((a, k)))
    analytics.track(analytics.LLMUsageEvent(provider="openai", model="m", llm_type="factory"))
    assert posted == []  # 외부로 아무것도 보내지 않는다


def test_build_evaluator_raises_the_output_limit_above_ragas_default(monkeypatch):
    import src.evaluation.ragas_faithfulness as rf

    seen = {}

    def fake_factory(model, provider="openai", client=None, **kwargs):
        seen.update(model=model, provider=provider, kwargs=kwargs)
        return FakeLLM([], [])

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setattr(rf, "llm_factory", fake_factory)
    rf.build_evaluator("gpt-6-luna")
    assert seen["kwargs"] == {"max_tokens": 4096} and seen["model"] == "gpt-6-luna"  # ragas 기본 1024보다 크다
    rf.build_evaluator("gpt-6-luna", max_output_tokens=2048)
    assert seen["kwargs"] == {"max_tokens": 2048}
    assert rf.tool_versions()["ragas_telemetry"] == "disabled"
