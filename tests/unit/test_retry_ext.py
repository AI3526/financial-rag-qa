"""되묻기·보류 재검색 (확장 브랜치 2단계): answered는 그대로, 아니면 reason으로 한 번 더 검색해 다시 생성한다."""
from src.api.pipeline import GenerationOutput, PipelineSettings
from src.api.retry_ext import RetryRagPipeline
from src.retrieval.dense import RetrievalResult


def chunk(i, text="내용"):
    return {"rank": i, "chunk_id": f"c{i}", "book_id": "B", "file_name": "B.pdf", "page": i, "text": text, "metadata": {}, "score": 1.0}


class FakeRetriever:
    def __init__(self):
        self.queries = []
        self.config = None

    def retrieve(self, q, k=5):
        self.queries.append(q)
        base = 1 if len(self.queries) == 1 else 20  # 두 번째 검색은 다른 청크
        return RetrievalResult(question=q, method="hybrid_rerank", k=k, chunks=[chunk(base + i) for i in range(k)], candidate_k=k, candidates={}, durations_ms={"total": 1.0})


class FakeGenerator:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def generate(self, question, sources):
        self.calls.append([s["chunk_id"] for s in sources])
        return GenerationOutput(parsed=self.outputs.pop(0), raw="{}", model_input=[], model_id="fake", usage={"input_tokens": 1, "output_tokens": 1})


def pipeline(outputs, final_k=12):
    r = FakeRetriever()
    g = FakeGenerator(outputs)
    return RetryRagPipeline(r, g, PipelineSettings(final_k=final_k), retrievers={"hybrid_rerank": r}), r, g


def out(status, reason="분기 매출이 필요합니다", ids=("S1",)):
    return {"status": status, "answer": "답", "reason": reason, "source_ids": list(ids) if status == "answered" else []}


def test_answered_is_returned_without_second_search():
    p, r, g = pipeline([out("answered")])
    o = p.run("rid", "질문", "hybrid_rerank")
    assert o.response["status"] == "answered" and len(r.queries) == 1 and len(g.calls) == 1
    assert o.record["retry"] == {"triggered": False, "first_status": "answered"}


def test_abstained_triggers_second_search_with_question_and_reason_and_merges_context():
    p, r, g = pipeline([out("abstained"), out("answered")])
    o = p.run("rid", "해온 질문", "hybrid_rerank")
    assert r.queries[1] == "해온 질문 분기 매출이 필요합니다"
    assert len(g.calls) == 2 and len(g.calls[1]) == 12  # 첫 문맥 8 + 새 청크 4
    assert g.calls[1][:8] == [f"c{i}" for i in range(1, 9)] and g.calls[1][8:] == ["c20", "c21", "c22", "c23"]
    assert o.response["status"] == "answered"
    assert o.record["retry"]["triggered"] and len(o.record["retry"]["added_chunks"]) == 4
    assert len(o.record["llm_calls"]) == 2


def pipeline_v(outputs, variant):
    r = FakeRetriever()
    g = FakeGenerator(outputs)
    return RetryRagPipeline(r, g, PipelineSettings(final_k=12), retrievers={"hybrid_rerank": r}, variant=variant), r, g


def test_broad_variant_retries_answered_with_missing_marker_but_base_does_not():
    partial = {"status": "answered", "answer": "매출은 480억원입니다. 다만 업무는 확인되지 않습니다.", "reason": "일부 확인", "source_ids": ["S1"]}
    p, r, g = pipeline_v([partial, out("answered")], "broad")
    p.run("rid", "질문", "hybrid_rerank")
    assert len(r.queries) == 2 and len(g.calls) == 2
    p, r, g = pipeline_v([partial], "base")
    p.run("rid", "질문", "hybrid_rerank")
    assert len(r.queries) == 1 and len(g.calls) == 1


def test_reason_variant_uses_reason_only_as_query():
    p, r, g = pipeline_v([out("abstained", reason="1분기 매출 필요"), out("answered")], "reason")
    p.run("rid", "해온 질문", "hybrid_rerank")
    assert r.queries[1] == "1분기 매출 필요"
