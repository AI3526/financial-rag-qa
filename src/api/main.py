"""서버 기본 조립. 실행: `uv run uvicorn --factory src.api.main:build_default_app`

import만으로 모델을 읽지 않도록 팩토리로 만든다. `OPENAI_API_KEY`와 `OPENAI_MODEL`이 있으면 gpt-6-luna 생성기를
쓰고, 없으면 생성기가 연결되지 않아 `/rag/query`가 정상 질문에도 `internal_error`를 돌려준다(답을 만들지 않음).
"""
from __future__ import annotations

import os

from src.config import ROOT, load_env


def build_default_app(prompt_name: str | None = None, use_langfuse: bool = True):
    """서버 조립. prompt_name으로 `prompts/{prompt_name}.json`을 고른다(기본 answer_v2).
    use_langfuse=False면 Langfuse에 기록하지 않는다(development 실험은 로컬에만, D-029)."""
    load_env()
    from src.api.app import create_app
    from src.api.pipeline import PipelineSettings, RagPipeline, UnavailableGenerator
    from src.api.recording import RunRecorder
    from src.observability.langfuse_client import get_langfuse
    from src.retrieval.dense import DenseRetriever

    settings = PipelineSettings()
    langfuse = get_langfuse() if use_langfuse else None
    if os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_MODEL"):
        from src.generation.generator import LangChainGenerator

        factory = None
        if langfuse is not None:
            from langfuse.langchain import CallbackHandler

            # 요청마다 새 핸들러. 현재 열린 `rag-query` Span 아래에 모델 호출(Generation)이 붙는다
            factory = lambda: [CallbackHandler()]  # noqa: E731
        from src.generation.prompts import load_prompt

        prompt = load_prompt(prompt_name) if prompt_name else load_prompt()
        generator = LangChainGenerator(settings, prompt=prompt, callbacks_factory=factory)
    else:
        generator = UnavailableGenerator()
    from src.generation.pricing import load_pricing

    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.rerank import CrossEncoderReranker, RerankRetriever
    from src.retrieval.sparse import SparseRetriever

    dense = DenseRetriever(langfuse=langfuse)
    # BM25는 같은 chunks.jsonl을 메모리에서 색인한다(토크나이저 whitespace 동결, D-035). hybrid_rerank는 아래에서 Reranker를 붙여 추가한다
    hybrid = HybridRetriever(dense, SparseRetriever(langfuse=langfuse), langfuse=langfuse)
    # Reranker는 모델을 한 번 읽어 둔다(BAAI/bge-reranker-v2-m3, D-037). 읽지 못하면 서버가 시작되지 않는다(조용히 빠지지 않음)
    reranked = RerankRetriever(hybrid, CrossEncoderReranker(), langfuse=langfuse)
    pipeline = RagPipeline(dense, generator, settings, pricing=load_pricing(),
                           retrievers={"dense": dense, "hybrid": hybrid, "hybrid_rerank": reranked}, langfuse=langfuse)
    return create_app(pipeline, langfuse=langfuse, recorder=RunRecorder(ROOT / "results"))
