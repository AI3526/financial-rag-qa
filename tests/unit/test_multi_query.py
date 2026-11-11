"""질문 분해·다중 질의 검색 (확장 브랜치 5a). 모델·네트워크 없이 확인한다."""
import json
from types import SimpleNamespace

from src.retrieval import query_decompose as qd
from src.retrieval.dense import RetrievalResult
from src.retrieval.hybrid import HybridConfig
from src.retrieval.multi_query import MultiQueryHybrid, merge_rrf


def _chunk(cid, rank, page=1):
    return {"chunk_id": cid, "book_id": "B", "file_name": "B.pdf", "page": page, "text": f"t-{cid}", "metadata": {}, "rank": rank, "score": 1.0 / rank}


def test_prompt_file_is_the_approved_text():
    spec = qd.load_prompt()
    assert spec["version"] == "query-decompose-v1" and spec["may_use_reference_data"] is False and len(spec["sha256"]) == 64
    assert "질문에 없는 회사, 기간, 수치, 기준을 추측해서 넣지 않습니다." in spec["system"]
    assert "질문에 없으면 전년 동기" not in spec["system"] and "비교 대상(질문에 기준이 없으면 전년 동기)을 별도 검색 질의로 만듭니다." in spec["system"]
    assert spec["user_template"] == "질문: {question}" and "가나전자" in spec["system"]
    assert qd.build_messages(spec, "Q?")[1] == ("human", "질문: Q?")


def test_parse_queries_drops_original_duplicates_and_blanks_and_caps_at_three():
    raw = json.dumps({"queries": ["원래 질문", "A", "  A ", "", "B", "C", "D"]}, ensure_ascii=False)
    assert qd.parse_queries(raw, "원래  질문") == (["A", "B", "C"], None)


def test_parse_queries_failures_fall_back_to_no_extra_queries():
    assert qd.parse_queries("not json", "q")[0] == [] and qd.parse_queries("not json", "q")[1]
    assert qd.parse_queries(json.dumps({"queries": "x"}), "q") == ([], "queries가 문자열 목록이 아님")
    assert qd.parse_queries(json.dumps({"queries": []}), "q") == ([], "queries가 비어 있음")
    assert qd.parse_queries(json.dumps({"queries": ["q"]}), "q") == ([], None)  # 원래 질문뿐이면 추가 질의 없음(정상)


def test_decompose_records_errors_instead_of_raising():
    class Boom:
        def invoke(self, _):
            raise RuntimeError("boom")

    class Ok:
        def invoke(self, _):
            return SimpleNamespace(content=json.dumps({"queries": ["x1", "x2"]}), usage_metadata={"input_tokens": 5, "output_tokens": 2}, response_metadata={})

    spec = qd.load_prompt()
    bad = qd.decompose(Boom(), spec, "q")
    assert bad["queries"] == [] and "RuntimeError" in bad["error"]
    good = qd.decompose(Ok(), spec, "q")
    assert good["queries"] == ["x1", "x2"] and good["error"] is None and good["usage"]["input_tokens"] == 5


def test_merge_rrf_rewards_chunks_found_by_several_queries_and_is_deterministic():
    a = [_chunk("x", 1), _chunk("y", 2), _chunk("z", 3)]
    b = [_chunk("w", 1), _chunk("y", 2)]
    merged = merge_rrf([a, b], rrf_k=60, top=3)
    # y는 두 목록에 있어 1위. x와 w는 점수가 같고(각 목록 1위) 순위도 같아 chunk_id 순(w, x)으로 정한다
    assert [c["chunk_id"] for c in merged] == ["y", "w", "x"] and [c["rank"] for c in merged] == [1, 2, 3]
    assert merge_rrf([a, b], 60, 3) == merged


class _FakeHybrid:
    config = HybridConfig(final_k=5, fused_top=20)

    def __init__(self):
        self.calls = []

    def describe(self):
        return {"fusion": "rrf"}

    def retrieve(self, question, k=None):
        self.calls.append((question, k))
        chunks = {"Q": [_chunk("a", 1), _chunk("b", 2)], "Q-sub": [_chunk("c", 1), _chunk("a", 2)]}[question]
        return RetrievalResult(question=question, method="hybrid", k=k, chunks=chunks)


def test_multi_query_hybrid_searches_original_plus_extra_queries_and_merges():
    fake = _FakeHybrid()
    mq = MultiQueryHybrid(fake, {"Q": ["Q-sub", "Q"]}, top=20)
    res = mq.retrieve("Q", k=3)
    assert [q for q, _ in fake.calls] == ["Q", "Q-sub"] and all(k == 20 for _, k in fake.calls)  # 원래 질문 + 추가 질의, 중복 질의는 한 번만
    assert [c["chunk_id"] for c in res.chunks] == ["a", "c", "b"] and res.candidates["queries"] == ["Q", "Q-sub"]
    assert len(res.candidates["fused"]) == 3 and mq.describe()["multi_query"] is True


def test_multi_query_hybrid_without_extra_queries_equals_plain_hybrid_order():
    fake = _FakeHybrid()
    res = MultiQueryHybrid(fake, {}, top=20).retrieve("Q", k=5)
    assert [c["chunk_id"] for c in res.chunks] == ["a", "b"] and len(fake.calls) == 1
