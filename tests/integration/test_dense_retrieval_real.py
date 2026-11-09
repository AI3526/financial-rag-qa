"""STEP 4 검색 테스트 (08 §4.4): 실제 Qdrant 컬렉션 + e5 모델로 질문 유형별 동작을 확인한다.

네트워크(Qdrant Cloud)와 e5 모델이 필요하다. 설정이 없으면 건너뛴다.
질문은 평가 데이터가 아니라 문서에 나오는 이름·용어로 직접 만들었다(정답·근거 원문은 쓰지 않는다).
"""
import json
from pathlib import Path

import pytest

from src.config import load_env

load_env()

from src.index import qdrant_store as qs  # noqa: E402
from src.retrieval.dense import DenseRetriever, InvalidQuery, RetrievalConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "data" / "processed" / "documents.jsonl"
CFG = RetrievalConfig()


def _available() -> bool:
    if qs.qdrant_mode() != "remote":
        return False
    try:
        return qs.collection_exists(qs.get_client(), CFG.collection)
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _available(), reason="Qdrant 컬렉션에 접속할 수 없습니다")


@pytest.fixture(scope="module")
def retriever(dense_retriever):
    return dense_retriever  # 세션에서 한 번만 읽은 모델 공유(conftest.py)


@pytest.fixture(scope="module")
def books():
    return {json.loads(l)["file_name"] for l in DOCS.read_text(encoding="utf-8").splitlines() if l.strip()}


# (유형, 질문, 상위 5개 청크 중 하나에 있어야 하는 용어)
TYPED = [
    ("회사명", "해온소재는 어떤 회사인가요?", "해온소재"),
    ("제품명", "루미센서 E는 어떤 제품인가요?", "루미센서"),
    ("제품명2", "리브필름 기본형의 재생 원료 비율은?", "리브필름"),
    ("기간", "2026년 상반기 실적을 알려 주세요.", "상반기"),
    ("수치", "영업이익은 얼마인가요?", "영업이익"),
    ("사업부문", "장비사업과 운영서비스사업은 어떻게 구분되나요?", "운영서비스"),
]


@pytest.mark.parametrize("kind,question,term", TYPED, ids=[t[0] for t in TYPED])
def test_typed_questions_return_ranked_traceable_chunks(retriever, books, kind, question, term):
    res = retriever.retrieve(question)
    assert len(res.chunks) == 5
    assert [c["rank"] for c in res.chunks] == [1, 2, 3, 4, 5]
    scores = [c["score"] for c in res.chunks]
    assert scores == sorted(scores, reverse=True)
    for c in res.chunks:  # 원본 PDF 페이지로 연결된다
        assert c["file_name"] in books and c["page"] >= 1 and c["text"].strip()
        assert c["book_id"] + ".pdf" == c["file_name"]
    assert any(term in c["text"] for c in res.chunks), f"{kind}: 상위 5개 청크에 '{term}'이 없음"


def test_insufficient_evidence_question_still_returns_top_k(retriever):
    # 자료에 답이 없는 질문이어도 검색은 정상 동작한다(근거 충분성 판단은 생성 단계의 몫)
    res = retriever.retrieve("해온솔루션의 2027년 1분기 확정 매출을 알려 주세요.")
    assert len(res.chunks) == 5 and all(c["score"] > 0 for c in res.chunks)


def test_unrelated_question_returns_chunks_with_lower_scores(retriever):
    related = retriever.retrieve("루미센서 E는 어떤 제품인가요?").chunks[0]["score"]
    unrelated = retriever.retrieve("오늘 저녁 메뉴로 김치찌개가 좋을까요?").chunks[0]["score"]
    assert unrelated < related  # 점수로 관련도가 구분된다 (임계값은 두지 않음)


def test_blank_question_is_rejected_before_searching(retriever):
    with pytest.raises(InvalidQuery):
        retriever.retrieve("   ")


def test_same_question_gives_same_ranking(retriever):
    a = retriever.retrieve("루미뷰 이용계약은 어떻게 되나요?")
    b = retriever.retrieve("루미뷰 이용계약은 어떻게 되나요?")
    assert [c["chunk_id"] for c in a.chunks] == [c["chunk_id"] for c in b.chunks]
