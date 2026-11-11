"""쪽 단위 선택 (확장 브랜치 4단계, EXTENSION_INTERMEDIATE_LOG).

순위가 매겨진 청크 목록에서 최종 k개를 고를 때 **같은 쪽의 청크를 쪽당 최대 `cap`개까지만** 허용한다. 정답을 쪽 단위로 세므로 같은 쪽 청크가
슬롯을 차지하면 다른 쪽을 넣을 자리가 줄어드는 문제를 줄이기 위한 것이다.
- cap=1: 쪽마다 점수가 가장 높은 청크 하나만 남긴다(서로 다른 쪽 k개). cap이 k 이상이면 입력 순서 그대로 상위 k개다(기존 방식).
- 건너뛴 청크는 버린다. 입력 순위를 바꾸지 않으며, 빈 자리는 순위상 다음 청크가 채운다. 결과의 `rank`는 1부터 다시 매긴다.
- 위험: 같은 쪽의 두 번째 청크가 답에 필요한 경우를 잃을 수 있다(cap=1). 그래서 cap=2도 함께 시험하고, 정답 문장이 최종 청크 안에 있는지 따로 잰다.
"""
from __future__ import annotations


def select_top_pages(chunks: list[dict], k: int = 5, cap: int = 1) -> list[dict]:
    if cap < 1:
        raise ValueError("cap은 1 이상이어야 합니다.")
    kept: list[dict] = []
    per_page: dict[tuple, int] = {}
    for c in chunks:
        key = (c["book_id"], c["page"])
        if per_page.get(key, 0) >= cap:
            continue
        per_page[key] = per_page.get(key, 0) + 1
        kept.append(c)
        if len(kept) == k:
            break
    return [{**c, "rank": i} for i, c in enumerate(kept, start=1)]
