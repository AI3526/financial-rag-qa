"""보안 비교(STEP 13)용 검색기 래퍼: 한 번 검색한 결과를 저장하고(`CapturingRetriever`), 같은 결과를 다시 돌려준다(`FixedRetriever`).

기본 프롬프트 실행에서 실제 검색을 하고 그 결과를 저장한 뒤, 방어 프롬프트 실행은 저장한 결과를 그대로 써서 두 실행 사이에 재검색하지 않는다.
- 평가 정답·공격 문장은 이 모듈에 들어오지 않는다.
"""
from __future__ import annotations

from dataclasses import replace

from src.retrieval.dense import RetrievalResult


class CapturingRetriever:
    """실제 검색기를 감싸 마지막 검색 결과를 `last`에 저장한다(결과는 그대로 돌려준다)."""

    def __init__(self, inner):
        self.inner = inner
        self.last: RetrievalResult | None = None
        self.config = getattr(inner, "config", None)
        self.langfuse = getattr(inner, "langfuse", None)

    def describe(self) -> dict:
        return self.inner.describe() if hasattr(self.inner, "describe") else {}

    def retrieve(self, question: str, k: int | None = None) -> RetrievalResult:
        result = self.inner.retrieve(question, k) if k is not None else self.inner.retrieve(question)
        self.last = result
        return result


class FixedRetriever:
    """미리 저장한 검색 결과를 그대로 돌려주고 새로 검색하지 않는다. 질문이 다르면 오류(다른 질문의 결과를 잘못 쓰지 않도록)."""

    def __init__(self, stored: RetrievalResult, describe: dict | None = None, config=None):
        self.stored = stored
        self._describe = dict(describe or {})
        self.config = config
        self.langfuse = None
        self.calls = 0

    def describe(self) -> dict:
        return {**self._describe, "reused_stored_result": True}

    def retrieve(self, question: str, k: int | None = None) -> RetrievalResult:
        if question != self.stored.question:
            raise ValueError("저장한 검색 결과의 질문과 다른 질문입니다.")
        self.calls += 1
        # 새로 검색하지 않았으므로 걸린 시간은 0으로 두고 후보 목록은 저장한 값을 그대로 쓴다
        return replace(self.stored, durations_ms={"total": 0.0, "reused": 1.0})
