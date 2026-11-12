"""질의의 분기 표기 병기 (확장 브랜치, 이 코퍼스 전용 규칙).

Intermediate 문서의 표 머리는 분기를 "2026 Q1"로 적는데 질문과 분해 질의는 "2026년 1분기"로 쓴다. 키워드 검색(BM25)이 둘을 같은 말로 보지 못해 표를 놓친다
(IF-DEV19: 질의에 "(2026 Q1)"을 병기하면 정답 쪽이 1위). 그래서 질의에 "연도년 N분기"가 있으면 "연도 QN"을 덧붙인다. LLM 호출이 없고 결과가 항상 같다.
- 다른 데이터셋에는 그대로 적용하지 않는다(문서의 표기를 확인해서 정한 규칙이다).
"""
from __future__ import annotations

import re

_QUARTER = re.compile(r"(\d{4})\s*년\s*([1-4])\s*분기")


def add_quarter_notation(query: str) -> str:
    """질의 안의 각 `연도년 N분기`마다 `연도 QN`을 한 번씩(중복 없이) 뒤에 덧붙인다. 해당 표현이 없으면 그대로 돌려준다."""
    extras = []
    for year, q in _QUARTER.findall(query):
        token = f"{year} Q{q}"
        if token not in query and token not in extras:
            extras.append(token)
    return f"{query} ({' '.join(extras)})" if extras else query
