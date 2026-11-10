"""기대 행동(expected_action)과 응답 상태(status) 비교 (STEP 9, 08 §13).

- 평가용 정보(expected_action, 정답)는 응답이 끝난 **뒤** 비교에만 쓴다. 서비스 요청에는 질문 문자열만 보낸다.
- 이것은 상태 일치를 보는 개발용 점검이다. 답변 내용의 정확성(필수 사실, 금지 주장)은 STEP 11의 Judge가 평가한다.
- 상태 이름보다 답한 범위와 보류 이유가 기대 행동에 맞는지가 중요하다(08 §9.3). 불일치는 사람이 응답을 읽어 판단한다.
"""
from __future__ import annotations

EXPECTED_STATUS = {"answer": "answered", "clarify": "needs_clarification", "abstain": "abstained"}


def request_body(question: dict, rag_method: str = "dense") -> dict:
    """서비스로 보내는 본문. 질문 ID와 질문 문자열만 담는다(정답·근거·기대 행동 제외)."""
    return {"question": question["question"], "rag_method": rag_method, "question_id": question["id"]}


def compare(question_id: str, expected_action: str, response: dict, http_status: int) -> dict:
    expected = EXPECTED_STATUS[expected_action]
    return {"question_id": question_id, "expected_action": expected_action, "expected_status": expected,
            "status": response.get("status"), "http_status": http_status,
            "match": response.get("status") == expected}


def summarize(rows: list[dict]) -> dict:
    """기대 행동별 일치 수. 오류 응답(error)은 일치로 세지 않고 따로 센다."""
    out: dict = {"total": len(rows), "match": sum(r["match"] for r in rows),
                 "errors": sum(r["status"] == "error" for r in rows), "by_action": {}}
    for action in EXPECTED_STATUS:
        sub = [r for r in rows if r["expected_action"] == action]
        out["by_action"][action] = {"total": len(sub), "match": sum(r["match"] for r in sub),
                                    "returned": {s: sum(r["status"] == s for r in sub)
                                                 for s in sorted({r["status"] for r in sub})}}
    return out
