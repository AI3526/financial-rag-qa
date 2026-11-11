"""질의별 최대 점수 재정렬 (확장 브랜치 (a))."""
from src.retrieval.multi_query import MaxOverQueriesReranker


class FakeScorer:
    model_id = "fake"
    revision = "r"

    def __init__(self):
        self.calls = []

    def score(self, q, texts):
        self.calls.append(q)
        table = {"원래": [0.9, 0.1], "분해": [0.2, 0.8]}
        return table[q][: len(texts)]


def test_takes_max_over_original_and_sub_queries():
    s = FakeScorer()
    r = MaxOverQueriesReranker(s, {"원래": ["분해"]})
    assert r.score("원래", ["a", "b"]) == [0.9, 0.8]
    assert s.calls == ["원래", "분해"]


def test_without_extra_queries_equals_original_scores():
    s = FakeScorer()
    r = MaxOverQueriesReranker(s, {})
    assert r.score("원래", ["a", "b"]) == [0.9, 0.1]
    assert r.score("원래", []) == []
