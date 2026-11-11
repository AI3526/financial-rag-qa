"""MRR·nDCG@k (확장 브랜치). 계산 규칙을 손으로 계산한 값과 대조한다."""
import math

import pytest

from src.evaluation.ranking_metrics import first_relevant_rank, ndcg_at_k, reciprocal_rank


def test_first_relevant_rank_and_reciprocal_rank():
    assert first_relevant_rank(["a", "b", "c"], {"c"}) == 3 and first_relevant_rank(["a"], {"z"}) is None
    assert reciprocal_rank(["a", "b", "c"], {"b", "c"}) == pytest.approx(0.5)
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0
    assert reciprocal_rank(["a", "b", "c"], {"c"}, k=2) == 0.0  # 상위 k 밖은 0


def test_empty_reference_is_not_applicable_never_zero():
    assert reciprocal_rank(["a"], set()) is None and ndcg_at_k(["a"], set()) is None


def test_ndcg_perfect_and_partial_and_duplicates():
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}) == pytest.approx(1.0)  # 정답을 맨 앞에 둠
    got = ndcg_at_k(["x", "a", "b"], {"a", "b"})
    assert got == pytest.approx((1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3)))
    assert ndcg_at_k(["a", "a", "a"], {"a", "b"}) == pytest.approx(1.0 / (1 + 1 / math.log2(3)))  # 같은 정답 중복은 한 번만


def test_ndcg_ideal_is_capped_by_k():
    ref = {f"p{i}" for i in range(10)}
    assert ndcg_at_k(["p0", "p1", "p2", "p3", "p4"], ref, k=5) == pytest.approx(1.0)
