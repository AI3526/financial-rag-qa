import pytest

from src.evaluation import retrieval_compare as rc

QUESTIONS = [{"id": "Q1", "question": "첫째 질문"}, {"id": "Q2", "question": "둘째 질문"}, {"id": "Q3", "question": "되묻기 질문"}]
ANSWERS = {
    "Q1": {"expected_action": "answer", "evidence_groups": [[{"book_id": "FB01", "page": 1, "quote": "x"}],
                                                            [{"book_id": "FB02", "page": 3, "quote": "y"}]]},
    "Q2": {"expected_action": "answer", "evidence_groups": [[{"book_id": "FB04", "page": 2, "quote": "z"}]]},
    "Q3": {"expected_action": "clarify", "evidence_groups": []},
}


def chunk(rank, book, page):
    return {"rank": rank, "chunk_id": f"{book}-p{page}-c1", "book_id": book, "page": page}


def test_retriever_receives_only_question_text_and_answers_are_read_after_all_searches():
    seen, order = [], []

    class Answers(dict):
        def __getitem__(self, key):
            order.append(("answer", key))
            return super().__getitem__(key)

    def retrieve(text):
        order.append(("search", text))
        seen.append(text)
        return [chunk(1, "FB01", 1)]

    rc.evaluate_retriever(retrieve, QUESTIONS, Answers(ANSWERS))
    assert seen == ["첫째 질문", "둘째 질문", "되묻기 질문"]  # 질문 문자열뿐
    assert all(kind == "search" for kind, _ in order[:3]) and all(kind == "answer" for kind, _ in order[3:])


def test_metrics_are_document_level_with_na_for_questions_without_evidence():
    results = {"첫째 질문": [chunk(1, "FB01", 1), chunk(2, "FB01", 9)], "둘째 질문": [chunk(1, "FB07", 1)], "되묻기 질문": []}
    out = rc.evaluate_retriever(lambda t: results[t], QUESTIONS, ANSWERS)
    s = out["summary"]
    assert (s["questions"], s["evaluated_questions"]) == (3, 2)  # 근거 없는 질문은 N/A로 제외
    assert s["hit_at_5"] == 0.5 and s["recall_at_5"] == 0.25  # Q1: hit 1 recall 0.5 / Q2: 0 0
    assert s["page_hit_at_5"] == 0.5 and s["page_recall_at_5"] == 0.25
    assert out["rows"][2]["hit"] is None and s["empty_results"] == 1


def summ(recall, hit):
    return {"recall_at_5": recall, "hit_at_5": hit}


def test_pick_by_rule_uses_recall_then_hit_then_the_preset_order():
    order = ["a", "b", "c"]
    assert rc.pick_by_rule({"a": summ(0.5, 0.9), "b": summ(0.6, 0.1)}, order) == ("b", "Recall@5")
    assert rc.pick_by_rule({"a": summ(0.5, 0.7), "b": summ(0.5, 0.9)}, order) == ("b", "Hit@5")
    assert rc.pick_by_rule({"a": summ(0.5, 0.7), "b": summ(0.5, 0.7), "c": summ(0.1, 0.1)}, ["b", "a", "c"]) == ("b", "동점(미리 정한 우선순위)")
    # 소수 9자리까지 같으면 동점
    assert rc.pick_by_rule({"a": summ(0.5, 0.7), "b": summ(0.5 + 1e-12, 0.7)}, order)[1] == "동점(미리 정한 우선순위)"
