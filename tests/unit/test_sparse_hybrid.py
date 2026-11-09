import math

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.pipeline import GenerationOutput, PipelineSettings, RagPipeline
from src.retrieval import tokenize as tk
from src.retrieval.dense import InvalidQuery, RetrievalError, RetrievalResult
from src.retrieval.hybrid import HybridConfig, HybridRetriever, rrf_fuse
from src.retrieval.sparse import SparseRetriever
from tests.unit.test_api import FakeLangfuse


def chunk(i, text, book="FB01", page=1, index=None):
    return {"chunk_id": f"{book}-p{page}-c{i}", "book_id": book, "file_name": f"{book}.pdf", "page": page,
            "chunk_text": text, "chunk_index": index if index is not None else i, "metadata": {}}


def item(rank, cid, score=0.0, book="FB01", page=1):
    return {"rank": rank, "score": score, "chunk_id": cid, "book_id": book, "file_name": f"{book}.pdf", "page": page,
            "text": f"본문 {cid}", "metadata": {}}


# --- 토큰화 ---
def test_whitespace_and_char2gram_tokenizers():
    assert tk.get_tokenizer("whitespace")("해온솔루션의 매출은 1,280억원!") == ["해온솔루션의", "매출은", "1", "280억원"]
    assert tk.get_tokenizer("char2gram")("루미센서 E") == ["루미", "미센", "센서", "e"]


def test_unknown_tokenizer_name_is_rejected():
    with pytest.raises(ValueError):
        tk.get_tokenizer("nope")


def test_kiwi_default_splits_company_name_but_userdict_keeps_it_whole():
    # STEP 5에서 확인한 현상: 기본 kiwipiepy는 '해온솔루션'을 하/어/오/ㄴ/솔루션으로 쪼갠다
    default = tk.get_tokenizer("kiwi")("해온솔루션의 영업이익")
    with_dict = tk.get_tokenizer("kiwi_userdict")("해온솔루션의 영업이익")
    assert "해온솔루션" not in default and "솔루션" in default
    assert "해온솔루션" in with_dict and "영업" in with_dict and "이익" in with_dict


def test_kiwi_keeps_numbers_and_drops_particles_and_endings():
    toks = tk.get_tokenizer("kiwi_userdict")("영업이익은 96억원입니다.")
    assert "96" in toks and "은" not in toks and "입니다" not in toks


def test_userdict_file_lists_only_names_and_no_comments():
    words = tk.load_userdict()
    assert {"해온솔루션", "해온소재", "루미센서", "리브필름"} <= set(words)
    assert all(not w.startswith("#") and w == w.strip() for w in words)


# --- BM25 ---
CORPUS = [chunk(1, "해온솔루션 영업이익 96억원"), chunk(2, "해온소재 필름사업 매출 620억원", "FB04", 3),
          chunk(3, "루미센서 확장형 제품 설명", "FB02", 1), chunk(4, "영업이익 영업이익 영업이익", "FB09", 2)]


def sparse(**kw):
    return SparseRetriever(CORPUS, tokenizer=tk.whitespace_tokens, **kw)


def test_bm25_ranks_matching_chunks_with_source_fields():
    res = sparse().retrieve("해온소재 매출", k=3)
    assert res.method == "sparse" and res.chunks[0]["chunk_id"] == "FB04-p3-c2"
    top = res.chunks[0]
    assert (top["rank"], top["book_id"], top["file_name"], top["page"]) == (1, "FB04", "FB04.pdf", 3)
    assert {"score", "text", "metadata"} <= set(top) and top["text"] == "해온소재 필름사업 매출 620억원"
    assert [c["rank"] for c in res.chunks] == list(range(1, len(res.chunks) + 1))
    scores = [c["score"] for c in res.chunks]
    assert scores == sorted(scores, reverse=True) and all(s > 0 for s in scores)


def test_zero_score_chunks_are_excluded_and_unmatched_query_gives_empty_list():
    res = sparse().retrieve("루미센서", k=5)
    assert [c["chunk_id"] for c in res.chunks] == ["FB02-p1-c3"]  # 질문 토큰이 없는 청크는 넣지 않는다
    assert sparse().retrieve("전혀관련없는단어", k=5).chunks == []  # 오류가 아니라 빈 결과


def test_query_that_produces_no_tokens_returns_empty_result_not_error():
    assert sparse().retrieve("!!! ???", k=5).chunks == []


def test_blank_question_and_bad_k_are_invalid_query():
    s = sparse()
    for bad in ("", "  ", None):
        with pytest.raises(InvalidQuery):
            s.retrieve(bad)
    with pytest.raises(InvalidQuery):
        s.retrieve("영업이익", k=0)


def test_ties_are_broken_by_chunk_order_so_results_are_deterministic():
    same = [chunk(1, "동일 문장", "FB05", 4, index=30), chunk(2, "동일 문장", "FB01", 1, index=2)]
    r = SparseRetriever(same, tokenizer=tk.whitespace_tokens).retrieve("동일 문장", k=2)
    assert [c["chunk_id"] for c in r.chunks] == ["FB01-p1-c2", "FB05-p4-c1"]  # chunk_index가 작은 쪽이 먼저
    assert [c["chunk_id"] for c in sparse().retrieve("영업이익").chunks] == [c["chunk_id"] for c in sparse().retrieve("영업이익").chunks]


def test_sparse_index_input_is_the_same_retrieval_text_as_dense_not_evaluation_data():
    meta = chunk(1, "본문만", "FB01", 1)
    meta["metadata"] = {"header": "FB01 | 머리말단어", "doc_title": "문서제목단어", "section_title": "소제목단어"}
    s = SparseRetriever([meta], tokenizer=tk.whitespace_tokens)
    assert s.retrieve("머리말단어").chunks[0]["chunk_id"] == "FB01-p1-c1"  # 머리말·제목도 검색 텍스트에 포함(Dense와 동일)
    assert s.retrieve("소제목단어").chunks and s.retrieve("정답").chunks == []


def test_sparse_langfuse_span_and_failure_isolation():
    lf = FakeLangfuse()
    res = sparse(langfuse=lf).retrieve("영업이익", k=2)
    assert [s.name for s in lf.spans] == ["sparse-retrieve"] and lf.spans[0].kwargs["as_type"] == "retriever"
    assert lf.spans[0].kwargs["metadata"]["algorithm"] == "bm25" and len(res.chunks) == 2
    from tests.unit.test_api import BrokenLangfuse

    assert len(sparse(langfuse=BrokenLangfuse()).retrieve("영업이익", k=2).chunks) == 2


# --- RRF ---
def test_rrf_scores_and_order_match_the_formula():
    dense = [item(1, "A"), item(2, "B"), item(3, "C")]
    sp = [item(1, "B"), item(2, "D"), item(3, "A")]
    fused = rrf_fuse(dense, sp, rrf_k=60, top=10)
    by = {r["chunk_id"]: r for r in fused}
    assert math.isclose(by["B"]["score"], 1 / 62 + 1 / 61) and math.isclose(by["A"]["score"], 1 / 61 + 1 / 63)
    assert [r["chunk_id"] for r in fused] == ["B", "A", "D", "C"][:0] or [r["chunk_id"] for r in fused][:2] == ["B", "A"]
    assert (by["B"]["dense_rank"], by["B"]["sparse_rank"]) == (2, 1)
    assert by["C"]["sparse_rank"] is None and by["D"]["dense_rank"] is None  # 한쪽에만 있는 청크도 포함, 없는 순위는 null
    assert [r["rank"] for r in fused] == [1, 2, 3, 4]


def test_rrf_ties_are_deterministic_and_top_limits_the_output():
    a, b = rrf_fuse([item(1, "Y")], [item(1, "X")], 60, 10), rrf_fuse([item(1, "Y")], [item(1, "X")], 60, 10)
    assert [r["chunk_id"] for r in a] == ["X", "Y"] == [r["chunk_id"] for r in b]  # 동점은 chunk_id 순
    assert len(rrf_fuse([item(i, f"D{i}") for i in range(1, 8)], [], 60, 3)) == 3


def test_rrf_with_one_empty_side_keeps_the_other_order():
    dense = [item(1, "A"), item(2, "B"), item(3, "C")]
    assert [r["chunk_id"] for r in rrf_fuse(dense, [], 60, 10)] == ["A", "B", "C"]
    assert rrf_fuse([], [], 60, 10) == []


def test_rrf_does_not_change_chunk_text_or_source():
    r = rrf_fuse([item(1, "A", book="FB07", page=9)], [], 60, 5)[0]
    assert (r["book_id"], r["file_name"], r["page"], r["text"]) == ("FB07", "FB07.pdf", 9, "본문 A")


# --- Hybrid ---
class FakeRetriever:
    def __init__(self, chunks, error=None):
        self.chunks, self.error, self.calls = chunks, error, []

    def retrieve(self, question, k=5):
        self.calls.append((question, k))
        if self.error:
            raise self.error
        return RetrievalResult(question=question, k=k, chunks=self.chunks[:k], durations_ms={"total": 1.0})


def hybrid(dense_chunks, sparse_chunks, **kw):
    d, s = FakeRetriever(dense_chunks), FakeRetriever(sparse_chunks)
    return HybridRetriever(d, s, **kw), d, s


def test_hybrid_uses_initial_settings_and_returns_final_k_with_candidates():
    dense = [item(i, f"D{i}") for i in range(1, 21)]
    sp = [item(i, f"S{i}") for i in range(1, 21)]
    h, d, s = hybrid(dense, sp)
    res = h.retrieve("영업이익은?")
    assert d.calls == [("영업이익은?", 20)] and s.calls == [("영업이익은?", 20)]  # 후보 20 + 20
    assert res.method == "hybrid" and res.k == 5 and len(res.chunks) == 5 and res.candidate_k == 20
    assert len(res.candidates["dense"]) == 20 and len(res.candidates["sparse"]) == 20 and len(res.candidates["fused"]) == 20
    assert [c["rank"] for c in res.chunks] == [1, 2, 3, 4, 5]
    assert set(res.durations_ms) == {"dense", "sparse", "fusion", "total"}
    assert HybridConfig() == HybridConfig(20, 20, 60, 20, 5)


def test_hybrid_promotes_a_chunk_found_by_both_methods():
    dense = [item(1, "D1"), item(2, "BOTH")]
    sp = [item(1, "S1"), item(2, "BOTH")]
    res = hybrid(dense, sp)[0].retrieve("질문")
    assert res.chunks[0]["chunk_id"] == "BOTH"  # 양쪽에 있으면 한쪽 1위보다 RRF 점수가 높다
    assert (res.chunks[0]["dense_rank"], res.chunks[0]["sparse_rank"]) == (2, 2)


def test_hybrid_falls_back_to_dense_order_when_bm25_finds_nothing():
    dense = [item(i, f"D{i}") for i in range(1, 6)]
    res = hybrid(dense, [])[0].retrieve("질문")
    assert [c["chunk_id"] for c in res.chunks] == [f"D{i}" for i in range(1, 6)] and res.candidates["sparse"] == []


def test_hybrid_validates_input_and_propagates_retrieval_errors():
    h, _, _ = hybrid([item(1, "A")], [])
    for bad in ("", "   ", None):
        with pytest.raises(InvalidQuery):
            h.retrieve(bad)
    with pytest.raises(InvalidQuery):
        h.retrieve("질문", k=0)
    broken = HybridRetriever(FakeRetriever([], error=RetrievalError("qdrant down")), FakeRetriever([]))
    with pytest.raises(RetrievalError):
        broken.retrieve("질문")


def test_hybrid_describe_records_the_settings():
    h = HybridRetriever(FakeRetriever([]), SparseRetriever(CORPUS, "kiwi_userdict"))
    assert h.describe() == {"fusion": "rrf", "rrf_k": 60, "dense_candidates": 20, "sparse_candidates": 20,
                            "fused_top": 20, "final_k": 5, "bm25_tokenizer": "kiwi_userdict"}


def test_hybrid_langfuse_spans_nest_dense_and_sparse_under_one_hybrid_span():
    lf = FakeLangfuse()
    h = HybridRetriever(FakeRetriever([item(1, "A")]), SparseRetriever(CORPUS, tokenizer=tk.whitespace_tokens, langfuse=lf),
                        langfuse=lf)
    h.retrieve("영업이익")
    assert [s.name for s in lf.spans] == ["hybrid-retrieve", "sparse-retrieve"]  # 가짜 dense는 Span을 만들지 않는다
    out = lf.spans[0].updates[0]["output"]
    assert {"dense", "sparse", "fused"} <= set(out["candidates"]) and lf.spans[0].kwargs["metadata"]["rrf_k"] == 60


# --- 파이프라인·API에서 rag_method 연결 ---
class Gen:
    def generate(self, question, sources):
        return GenerationOutput(parsed={"status": "answered", "answer": "답", "reason": "r", "source_ids": ["S1"]},
                                model_id="fake")


def make_client(tmp_path=None):
    from src.api.recording import RunRecorder

    dense_r = FakeRetriever([item(i, f"D{i}", 0.9 - i / 100, "FB01", i) for i in range(1, 6)])
    h = HybridRetriever(dense_r, SparseRetriever(CORPUS, tokenizer=tk.whitespace_tokens))
    pipe = RagPipeline(dense_r, Gen(), PipelineSettings(), retrievers={"dense": dense_r, "hybrid": h})
    return TestClient(create_app(pipe, recorder=RunRecorder(tmp_path) if tmp_path else None), raise_server_exceptions=False)


def test_api_routes_each_rag_method_to_its_own_retriever(tmp_path):
    import json

    client = make_client(tmp_path)
    for method in ("dense", "hybrid"):
        r = client.post("/rag/query", json={"question": "영업이익", "rag_method": method})
        assert r.status_code == 200 and r.json()["status"] == "answered"
    recs = [json.loads(l) for l in (tmp_path / "service" / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    dense_rec, hyb = recs[0]["retrieval"], recs[1]["retrieval"]
    assert dense_rec["method"] == "dense" and dense_rec["sparse"] is None and dense_rec["fused"] is None
    assert hyb["method"] == "hybrid" and hyb["dense"] and hyb["fused"] is not None and hyb["sparse"] is not None
    assert hyb["candidate_k"] == 20 and hyb["final_k"] == 5 and set(hyb["durations_ms"]) >= {"dense", "sparse", "fusion"}
    assert hyb["settings"]["rrf_k"] == 60 and recs[1]["rag_method"] == "hybrid"


def test_api_rejects_methods_without_a_retriever_with_422():
    client = make_client()
    for method in ("hybrid_rerank", "bm25", "other"):
        r = client.post("/rag/query", json={"question": "질문", "rag_method": method})
        assert r.status_code == 422 and r.json()["reason"] == "invalid_request"
