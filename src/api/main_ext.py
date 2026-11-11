"""확장 브랜치용 서비스 조립 (extension/intermediate, DECISIONS D-047): numpy 정확 검색, Qdrant·Langfuse 없음.

기존 `main.build_default_app`과 같은 파이프라인(`RagPipeline`)·생성기·출력 검증·run 레코드를 쓰되 검색기만 확장 구성으로 바꾼다.
- Dense는 numpy 정확 검색, Hybrid는 Dense + BM25(공백 토큰) RRF, Hybrid+Rerank는 bge-reranker-v2-m3. 후보 수는 `cands`로 정한다.
- 선택 옵션: `queries_lookup`(질문 분해 결과, 다중 질의), `routes_lookup`(문서 라우팅 결과). 둘을 함께 쓰지 않는다.
- 생성 설정·프롬프트(기본 answer_v2)·문맥 한도·최종 k는 Basic과 같다(`PipelineSettings` 기본값). 답변 단계 평가에서 검색 구성만 다르게 하려는 것이다.
- run 레코드의 `versions`에는 데이터셋(intermediate), 청킹 버전, 임베딩 버전이 들어간다.
"""
from __future__ import annotations

from pathlib import Path

from src.config import ROOT, load_env


def build_ext_app(*, processed_dir: Path, index_name: str, chunking_version: str, cands: tuple[int, int, int, int] = (20, 20, 20, 20),
                  prompt_name: str | None = None, queries_lookup: dict | None = None, routes_lookup: dict | None = None, final_k: int | None = None):
    """(app, pipeline)을 돌려준다. cands = (Dense 후보, BM25 후보, fused_top, Rerank 후보)."""
    load_env()
    from src.api.app import create_app
    from src.api.pipeline import PipelineSettings, RagPipeline
    from src.api.recording import RunRecorder
    from src.generation.generator import LangChainGenerator
    from src.generation.pricing import load_pricing
    from src.generation.prompts import load_prompt
    from src.index.embedding import Embedder
    from src.index.numpy_store import NumpyIndex
    from src.retrieval.dense import RetrievalConfig
    from src.retrieval.dense_np import NumpyDenseRetriever
    from src.retrieval.doc_routed import DocRoutedHybrid
    from src.retrieval.hybrid import HybridConfig, HybridRetriever
    from src.retrieval.multi_query import MultiQueryHybrid
    from src.retrieval.rerank import CrossEncoderReranker, RerankConfig, RerankRetriever
    from src.retrieval.sparse import SparseRetriever, load_chunks

    if queries_lookup is not None and routes_lookup is not None:
        raise ValueError("질문 분해와 문서 라우팅은 함께 쓰지 않는다(한 번에 하나만 바꾼다).")
    d, s, f, r = cands
    settings = PipelineSettings() if final_k is None else PipelineSettings(final_k=final_k)  # final_k: 확장 실험용 최종 청크 수(기본은 Basic과 같은 5)
    cfg = RetrievalConfig(dataset_id="intermediate", dataset_version="intermediate-v1", chunking_version=chunking_version)
    index = NumpyIndex.load(index_name)
    dense = NumpyDenseRetriever(index, cfg, embedder=Embedder(cfg.spec))
    hybrid = HybridRetriever(dense, SparseRetriever(chunks=load_chunks(processed_dir / "chunks.jsonl")),
                             HybridConfig(dense_candidates=d, sparse_candidates=s, fused_top=f, final_k=settings.final_k))
    if queries_lookup is not None:
        hybrid = MultiQueryHybrid(hybrid, queries_lookup, top=f)
    if routes_lookup is not None:
        hybrid = DocRoutedHybrid(hybrid, routes_lookup, top=f)
    reranked = RerankRetriever(hybrid, CrossEncoderReranker(), RerankConfig(candidates=r, final_k=settings.final_k))
    prompt = load_prompt(prompt_name) if prompt_name else load_prompt()
    generator = LangChainGenerator(settings, prompt=prompt, callbacks_factory=None)
    pipeline = RagPipeline(dense, generator, settings, pricing=load_pricing(), retrievers={"dense": dense, "hybrid": hybrid, "hybrid_rerank": reranked}, langfuse=None)
    return create_app(pipeline, langfuse=None, recorder=RunRecorder(ROOT / "results")), pipeline
