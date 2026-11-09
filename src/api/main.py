"""서버 기본 조립. 실행: `uv run uvicorn --factory src.api.main:build_default_app`

import만으로 모델을 읽지 않도록 팩토리로 만든다. STEP 8 전에는 생성기가 연결되지 않아
`/rag/query`가 정상 질문에도 `internal_error`(생성 모델 미연결)를 돌려준다.
"""
from __future__ import annotations

from src.config import ROOT, load_env


def build_default_app():
    load_env()
    from src.api.app import create_app
    from src.api.pipeline import PipelineSettings, RagPipeline, UnavailableGenerator
    from src.api.recording import RunRecorder
    from src.observability.langfuse_client import get_langfuse
    from src.retrieval.dense import DenseRetriever

    langfuse = get_langfuse()
    pipeline = RagPipeline(DenseRetriever(langfuse=langfuse), UnavailableGenerator(), PipelineSettings())
    return create_app(pipeline, langfuse=langfuse, recorder=RunRecorder(ROOT / "results"))
