"""RAGAS Faithfulness 실행 (STEP 11, D-039). **ragas가 설치된 격리 환경에서만 import한다.**

    uv run --with ragas --with "langchain-community<0.4" python scripts/run_ragas.py …

메인 환경의 `openai` 3.46.0을 바꾸지 않으려고 이렇게 실행한다(ragas 0.4.3은 이 환경에서 `openai` 3.3.0과 함께 설치된다).

ragas는 기본으로 사용 통계(익명 ID, ragas 버전, 모델 이름, 지표 이름, 호출 횟수)를 외부(t.explodinggradients.com)로 보낸다.
질문·답변·문맥 텍스트는 포함되지 않지만 우리는 보내지 않기로 해서 ragas를 import하기 **전에** `RAGAS_DO_NOT_TRACK=true`를 설정한다.

`DetailedFaithfulness`는 ragas의 `Faithfulness`를 상속해 두 가지만 바꾼다.
1. 1단계(주장 쪼개기) 프롬프트를 `prompts/faithfulness_statement_v1.json`으로 교체한다. 2단계(판정)는 ragas 기본값 그대로다.
2. 2단계가 주장마다 쓰는 이유(`reason`)와 판정(`verdict`)을 버리지 않고 결과에 담는다.
ragas의 비공개 메서드(`_create_statements`, `_create_verdicts`, `_compute_score`)를 쓰므로 ragas 버전을 고정·기록해야 한다.
평가 호출의 토큰 사용량은 연결 단계에서 세어 서비스 답변 생성 사용량과 분리한다.
"""
from __future__ import annotations

import os

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")  # ragas를 import하기 전에 설정해야 통계 전송이 꺼진다

import asyncio  # noqa: E402
import contextvars  # noqa: E402
import time  # noqa: E402
from importlib.metadata import version  # noqa: E402
from typing import Any  # noqa: E402

import httpx  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics.collections import Faithfulness  # noqa: E402
from ragas.metrics.collections.faithfulness.util import (StatementGeneratorInput, StatementGeneratorOutput,  # noqa: E402
                                                          StatementGeneratorPrompt)

from src.evaluation.faithfulness import build_detail, failed_result, load_statement_prompt  # noqa: E402

# ragas 기본 출력 한도는 1024토큰인데 gpt-6-luna의 추론 토큰이 이를 넘어 평가가 실패한 사례가 있어(D-039) 4096으로 올린다.
# `llm_factory`의 `max_tokens`는 OpenAI 추론 모델에는 `max_completion_tokens`로 전달된다(확인함).
DEFAULT_MAX_OUTPUT_TOKENS = 4096

_usage: contextvars.ContextVar[dict | None] = contextvars.ContextVar("scoring_usage", default=None)


def make_statement_prompt(spec: dict) -> StatementGeneratorPrompt:
    """승인된 프롬프트 파일의 지시문과 예시로 1단계 프롬프트 객체를 만든다."""
    examples = [(StatementGeneratorInput(question=e["question"], answer=e["answer"]),
                 StatementGeneratorOutput(statements=e["statements"])) for e in spec["examples"]]
    cls = type("StatementGeneratorPromptV1", (StatementGeneratorPrompt,), {"instruction": spec["instruction"], "examples": examples})
    return cls()


class DetailedFaithfulness(Faithfulness):
    def __init__(self, llm, spec: dict | None = None, **kwargs):
        super().__init__(llm=llm, **kwargs)
        self.prompt_spec = spec or load_statement_prompt()
        self.statement_generator_prompt = make_statement_prompt(self.prompt_spec)

    async def evaluate(self, question: str, answer: str, contexts: list[str]) -> dict:
        """답변 1건을 채점해 주장별 이유·판정과 사용량을 담은 dict를 돌려준다. 예외는 평가 불가로 기록한다(밖으로 던지지 않음)."""
        usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cached_input_tokens": 0}
        token = _usage.set(usage)
        started = time.perf_counter()
        try:
            statements = await self._create_statements(question, answer)
            if not statements:
                detail = build_detail([], [])
            else:
                verdicts = await self._create_verdicts(statements, "\n".join(contexts))
                detail = build_detail(statements, [v.model_dump() for v in verdicts.statements])
                ragas_score = self._compute_score(verdicts)  # ragas 자체 계산과 같은지 교차 확인용
                detail["ragas_score"] = None if ragas_score != ragas_score else float(ragas_score)
            detail["statements_generated"] = len(statements)
        except Exception as exc:  # noqa: BLE001 - 평가 실패는 점수가 아니라 평가 불가로 남긴다
            detail = failed_result(exc)
        finally:
            _usage.reset(token)
        detail["seconds"] = round(time.perf_counter() - started, 2)
        detail["usage"] = usage
        return detail


async def _count_usage(resp: httpx.Response) -> None:
    await resp.aread()
    acc = _usage.get()
    if acc is None:
        return
    try:
        u = resp.json().get("usage") or {}
    except Exception:  # noqa: BLE001
        return
    acc["calls"] += 1
    acc["input_tokens"] += u.get("prompt_tokens") or 0
    acc["output_tokens"] += u.get("completion_tokens") or 0
    acc["reasoning_tokens"] += (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    acc["cached_input_tokens"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0


def build_evaluator(model: str | None = None, spec: dict | None = None,
                    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> DetailedFaithfulness:
    """평가용 LLM(기본 `OPENAI_MODEL`, 서비스와 같은 gpt-6-luna)으로 채점기를 만든다. 일시 오류는 SDK가 2번까지 다시 시도한다."""
    client = AsyncOpenAI(http_client=httpx.AsyncClient(event_hooks={"response": [_count_usage]}, timeout=180), max_retries=2)
    llm = llm_factory(model or os.environ["OPENAI_MODEL"], provider="openai", client=client, max_tokens=max_output_tokens)
    return DetailedFaithfulness(llm, spec)


def tool_versions() -> dict[str, str]:
    return {"ragas": version("ragas"), "openai_in_eval_env": version("openai"), "langchain-community": version("langchain-community"),
            "ragas_telemetry": "disabled" if os.environ.get("RAGAS_DO_NOT_TRACK", "").lower() == "true" else "ENABLED"}


async def evaluate_many(metric: DetailedFaithfulness, items: list[dict], passes: int = 1, concurrency: int = 6,
                        on_done: Any = None) -> list[dict]:
    """items: {request_id, rag_method, question, answer, contexts, …}. (답변 × 횟수)마다 채점하고 입력 순서대로 결과를 돌려준다."""
    sem = asyncio.Semaphore(concurrency)

    async def one(item: dict, n: int) -> dict:
        async with sem:
            detail = await metric.evaluate(item["question"], item["answer"], item["contexts"])
        row = {"request_id": item["request_id"], "question_id": item.get("question_id"), "rag_method": item["rag_method"],
               "service_status": item.get("service_status"), "pass": n, **detail}
        if on_done is not None:
            on_done(item, row)
        return row

    return list(await asyncio.gather(*[one(it, n) for it in items for n in range(1, passes + 1)]))
