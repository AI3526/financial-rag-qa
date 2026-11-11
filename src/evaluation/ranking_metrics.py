"""순위 민감 검색 지표 MRR·nDCG@k (확장 브랜치, 10_ADDITIONAL_METRICS §2.2·§2.3).

- 입력은 검색된 청크를 순위대로 놓은 단위 목록(문서 단위는 book_id, 쪽 단위는 (book_id, page))과 정답 집합이다.
- 관련성은 이진이다. 같은 정답 단위를 가리키는 청크가 여러 개여도 **처음 나온 청크만** 센다(중복 이득 없음).
- 정답 집합이 비어 있으면 None(해당 없음)이다. 0점이 아니다.
- MRR: 처음 관련 단위가 나온 순위의 역수(상위 k 안에 없으면 0). nDCG@k: 이진 이득, 이상적 순서는 min(정답 수, k)개를 맨 앞에 둔 것.
"""
from __future__ import annotations

import math
from typing import Hashable, Sequence


def first_relevant_rank(units: Sequence[Hashable], reference: set) -> int | None:
    """처음 관련 단위가 나온 순위(1부터). 없으면 None."""
    for i, u in enumerate(units, start=1):
        if u in reference:
            return i
    return None


def reciprocal_rank(units: Sequence[Hashable], reference: set, k: int | None = None) -> float | None:
    if not reference:
        return None
    window = list(units)[:k] if k else list(units)
    r = first_relevant_rank(window, reference)
    return 0.0 if r is None else 1.0 / r


def ndcg_at_k(units: Sequence[Hashable], reference: set, k: int = 5) -> float | None:
    if not reference:
        return None
    seen: set = set()
    dcg = 0.0
    for i, u in enumerate(list(units)[:k], start=1):
        if u in reference and u not in seen:
            seen.add(u)
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(reference), k) + 1))
    return dcg / ideal
