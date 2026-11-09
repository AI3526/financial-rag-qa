"""서버 기본 조립. 실행: `uv run uvicorn --factory src.api.main:build_default_app`

import만으로 모델을 읽지 않도록 팩토리로 만든다. `OPENAI_API_KEY`와 `OPENAI_MODEL`이 있으면 gpt-6-luna 생성기를
쓰고, 없으면 생성기가 연결되지 않아 `/rag/query`가 정상 질문에도 `internal_error`를 돌려준다(답을 만들지 않음).
"""
from __future__ import annotations

import os

from src.config import ROOT, load_env


def build_default_app():
    load_env()
    from src.api.app import create_app
    from src.api.pipeline import PipelineSettings, RagPipeline, UnavailableGenerator
    from src.api.recording import RunRecorder
    from src.observability.langfuse_client import get_langfuse
    from src.retrieval.dense import DenseRetriever

    settings = PipelineSettings()
    langfuse = get_langfuse()
    if os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_MODEL"):
        from src.generation.generator import LangChainGenerator

        factory = None
        if langfuse is not None:
            from langfuse.langchain import CallbackHandler

            # 요청마다 새 핸들러. 현재 열린 `rag-query` Span 아래에 모델 호출(Generation)이 붙는다
            factory = lambda: [CallbackHandler()]  # noqa: E731
        generator = LangChainGenerator(settings, callbacks_factory=factory)
    else:
        generator = UnavailableGenerator()
    from src.generation.pricing import load_pricing

    pipeline = RagPipeline(DenseRetriever(langfuse=langfuse), generator, settings, pricing=load_pricing())
    return create_app(pipeline, langfuse=langfuse, recorder=RunRecorder(ROOT / "results"))
