"""통합 테스트가 모델을 세션당 한 번만 읽도록 공유한다.

각 테스트 파일이 e5 임베딩 모델(GPU 약 2.2GB)과 Reranker(약 2.3GB)를 따로 읽으면 8GB GPU의 메모리가 차서 전체 실행이
72초에서 600초 이상으로 느려졌다(STEP 6에서 확인). 모델은 이 fixture 하나로 공유한다.
"""
import pytest

from src.config import load_env

load_env()


@pytest.fixture(scope="session")
def dense_retriever():
    from src.retrieval.dense import DenseRetriever

    return DenseRetriever()


@pytest.fixture(scope="session")
def sparse_retriever():
    from src.retrieval.sparse import SparseRetriever

    return SparseRetriever()


@pytest.fixture(scope="session")
def shared_reranker():
    from src.retrieval.rerank import CrossEncoderReranker

    return CrossEncoderReranker()
