"""검색기를 development 질문으로 비교하는 공통 도구 (STEP 5, 08 §9.3).

- 검색기에는 질문 문자열만 넣는다. `evidence_groups`는 검색이 끝난 뒤 지표 계산에만 쓴다.
- 지표는 `book_id`(문서) 단위(02 §5.3, 공식)이고 `(book_id, page)` 단위는 참고다.
- 근거가 없는 질문(clarify 등)은 N/A로 제외한다.
"""
from __future__ import annotations

from typing import Callable

from src.evaluation.retrieval_metrics import mean_ignoring_none, question_metrics


def evaluate_retriever(retrieve: Callable[[str], list[dict]], questions: list[dict], answers: dict[str, dict]) -> dict:
    """retrieve(question_text) -> 청크 목록. 질문별 결과와 요약을 돌려준다.

    questions: 질문 파일(id, question만). answers: id -> 정답 항목(evidence_groups 포함), 검색 뒤에만 사용한다.
    """
    retrieved = {q["id"]: retrieve(q["question"]) for q in questions}  # 검색이 모두 끝난 뒤에 정답을 읽는다
    rows = []
    for q in questions:
        chunks = retrieved[q["id"]]
        m = question_metrics(chunks, answers[q["id"]]["evidence_groups"])
        rows.append({"question_id": q["id"], "expected_action": answers[q["id"]]["expected_action"], **m,
                     "retrieved": [{"chunk_id": c["chunk_id"], "book_id": c["book_id"], "page": c["page"],
                                    "rank": c["rank"]} for c in chunks]})
    summary = {
        "questions": len(rows), "evaluated_questions": sum(r["hit"] is not None for r in rows),
        "hit_at_5": mean_ignoring_none([r["hit"] for r in rows]),
        "recall_at_5": mean_ignoring_none([r["recall"] for r in rows]),
        "page_hit_at_5": mean_ignoring_none([r["page_hit"] for r in rows]),
        "page_recall_at_5": mean_ignoring_none([r["page_recall"] for r in rows]),
        "empty_results": sum(1 for r in rows if not r["retrieved"]),
    }
    return {"summary": summary, "rows": rows}


def pick_by_rule(summaries: dict[str, dict], tie_order: list[str]) -> tuple[str, str]:
    """사전 고정 규칙: Recall@5가 큰 쪽 → Hit@5가 큰 쪽 → 그래도 같으면 tie_order에서 앞선 쪽.

    같다는 판단은 소수 9자리 기준이다. (승자 이름, 결정 기준)을 돌려준다.
    """
    best = list(summaries)
    for field, label in (("recall_at_5", "Recall@5"), ("hit_at_5", "Hit@5")):
        top = max(round(summaries[n][field], 9) for n in best)
        best = [n for n in best if round(summaries[n][field], 9) == top]
        if len(best) == 1:
            return best[0], label
    return min(best, key=tie_order.index), "동점(미리 정한 우선순위)"
