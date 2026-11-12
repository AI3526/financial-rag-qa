"""분기 표기 병기와 분해 계획 자리 배정 (확장 브랜치)."""
from src.retrieval.dense import RetrievalResult
from src.retrieval.period_notation import add_quarter_notation
from src.retrieval.plan_slot import PlanSlotRetriever


def test_quarter_notation_appends_once_per_quarter_and_leaves_others_unchanged():
    assert add_quarter_notation("해온솔루션 2026년 1분기 별도 매출") == "해온솔루션 2026년 1분기 별도 매출 (2026 Q1)"
    assert add_quarter_notation("2026년 1분기와 2026년 2분기 매출") == "2026년 1분기와 2026년 2분기 매출 (2026 Q1 2026 Q2)"
    assert add_quarter_notation("해온솔루션 상반기 별도 매출") == "해온솔루션 상반기 별도 매출"
    assert add_quarter_notation("해온솔루션 2026 Q1 매출 2026년 1분기") == "해온솔루션 2026 Q1 매출 2026년 1분기"  # 이미 표기가 있으면 덧붙이지 않음


def chunk(i, prefix="c"):
    return {"rank": i, "chunk_id": f"{prefix}{i}", "book_id": "B", "file_name": "B.pdf", "page": i, "text": "t", "metadata": {}, "score": 1.0}


class FakeRerank:
    class config:
        final_k = 12

    def describe(self):
        return {}

    def retrieve(self, q, k=5):
        return RetrievalResult(question=q, method="hybrid_rerank", k=k, chunks=[chunk(i, "o") for i in range(1, k + 1)], candidate_k=k, candidates={}, durations_ms={"total": 1.0})


class FakeDense:
    def __init__(self):
        self.queries = []

    def retrieve(self, q, k=5):
        self.queries.append(q)
        return RetrievalResult(question=q, method="dense", k=k, chunks=[chunk(i, "d") for i in range(1, k + 1)], candidate_k=k, candidates={}, durations_ms={"total": 1.0})


class FakeSparse(FakeDense):
    def retrieve(self, q, k=5):
        self.queries.append(q)
        return RetrievalResult(question=q, method="sparse", k=k, chunks=[chunk(i, "b") for i in range(1, k + 1)], candidate_k=k, candidates={}, durations_ms={"total": 1.0})


class FakeHybrid:
    def __init__(self):
        self.queries = []
        self.dense, self.sparse = FakeDense(), FakeSparse()
        from src.retrieval.hybrid import HybridConfig

        self.config = HybridConfig()

    def retrieve(self, q, k=5):
        self.queries.append(q)
        n = len(self.queries)
        return RetrievalResult(question=q, method="hybrid", k=k, chunks=[chunk(i, f"s{n}_") for i in range(1, k + 1)], candidate_k=k, candidates={}, durations_ms={"total": 1.0})


def test_quarter_notation_goes_to_keyword_search_only_and_slots_are_reserved():
    h = FakeHybrid()
    r = PlanSlotRetriever(FakeRerank(), h, {"질문": ["해온 2026년 1분기 매출", "해온 상반기 매출"]}, pool=40)
    out = r.retrieve("질문", k=12)
    assert h.dense.queries == ["해온 2026년 1분기 매출"]  # Dense는 원래 질의
    assert h.sparse.queries == ["해온 2026년 1분기 매출 (2026 Q1)"]  # BM25만 분기 표기 병기
    assert h.queries == ["해온 상반기 매출"]  # 분기 표기가 없는 질의는 일반 Hybrid
    ids = [c["chunk_id"] for c in out.chunks]
    assert len(ids) == 12 and len(set(ids)) == 12 and sum(1 for c in out.chunks if c["slot_source"] != "original") == 4  # 질의당 2개
    h2 = FakeHybrid()
    r2 = PlanSlotRetriever(FakeRerank(), h2, {}, pool=40)
    out2 = r2.retrieve("질문", k=12)
    assert [c["chunk_id"] for c in out2.chunks] == [f"o{i}" for i in range(1, 13)] and h2.queries == []
