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


def quote_coverage(chunks: list[dict], evidence_groups: list[list[dict]]) -> tuple[int | None, float | None]:
    """정답 근거 문장(인용문)이 최종 청크 본문 안에 온전히 들어 있는 정도: (하나라도 있으면 1 아니면 0, 들어 있는 인용문의 비율).

    쪽이 맞아도 정답 문장이 든 청크가 빠지면 답변에 근거가 전달되지 않는다. 쪽 단위 지표가 놓치는 이 부분을 잰다(공백 제외 비교).
    인용문이 없으면 (None, None)이다. 청크는 `text` 필드에 본문을 가진 딕셔너리다.
    """
    import re

    quotes = {re.sub(r"\s+", "", e["quote"]) for g in evidence_groups for e in g if e.get("quote")}
    if not quotes:
        return None, None
    texts = [re.sub(r"\s+", "", c["text"]) for c in chunks]
    found = sum(1 for q in quotes if any(q in t for t in texts))
    return (1 if found else 0), found / len(quotes)
