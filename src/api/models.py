"""API 요청·응답 모델 (STEP 7, 09_DATA_SCHEMA.md §8).

요청 본문은 `{question, rag_method, question_id?}`뿐이다. 그 밖의 필드(예: 평가 정답)는 무시하고 서비스 입력으로
전달하지 않는다. 빈 질문과 지원하지 않는 `rag_method`는 파이프라인이 `invalid_request`(422)로 처리한다.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str
    rag_method: str
    question_id: str | None = None


class Citation(BaseModel):
    source_id: str
    book_id: str
    file_name: str
    page: int


class QueryResponse(BaseModel):
    request_id: str
    status: Literal["answered", "needs_clarification", "abstained", "error"]
    answer: str
    reason: str
    citations: list[Citation]
