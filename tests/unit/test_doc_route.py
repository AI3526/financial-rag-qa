"""문서 단위 라우팅과 문서 한정 검색 (확장 브랜치 방법 A). 모델·네트워크 없이 확인한다."""
import json
from types import SimpleNamespace

import numpy as np

from src.index.numpy_store import NumpyIndex
from src.retrieval import doc_route as dr
from src.retrieval.dense import RetrievalResult
from src.retrieval.doc_routed import DocRoutedHybrid
from src.retrieval.hybrid import HybridConfig


def _page(book, page, title):
    text = f"{book} | 관리실\n{title}\n소제목\n본문\n{page} / 3"
    return {"book_id": book, "page": page, "text": text}


def test_prompt_file_is_the_approved_text_with_a_catalog_placeholder():
    spec = dr.load_prompt()
    assert spec["version"] == "doc-route-v1" and spec["may_use_reference_data"] is False and len(spec["sha256"]) == 64
    assert "{catalog}" in spec["system"] and "목록에 없는 문서 번호를 만들지 않습니다." in spec["system"] and spec["user_template"] == "질문: {question}"
    msgs = dr.build_messages(spec, "IF01 | 제목", "Q?")
    assert "IF01 | 제목" in msgs[0][1] and "{catalog}" not in msgs[0][1] and '{"doc_ids"' in msgs[0][1] and msgs[1] == ("human", "질문: Q?")


def test_catalog_uses_each_documents_first_page_title():
    pages = [_page("IF02", 2, "둘째 쪽 제목"), _page("IF02", 1, "해온솔루션 보고서"), _page("IF01", 1, "기준 사전")]
    cat = dr.build_catalog(pages)
    assert cat == [("IF01", "기준 사전"), ("IF02", "해온솔루션 보고서")] and dr.catalog_text(cat) == "IF01 | 기준 사전\nIF02 | 해온솔루션 보고서"


def test_parse_doc_ids_drops_unknown_ids_duplicates_and_caps_at_four():
    valid = {f"IF{i:02d}" for i in range(1, 9)}
    raw = json.dumps({"doc_ids": ["IF01", " if02 ", "IF01", "IF99", "IF03", "IF04", "IF05"]})
    assert dr.parse_doc_ids(raw, valid) == (["IF01", "IF02", "IF03", "IF04"], 1, None)
    assert dr.parse_doc_ids(json.dumps({"doc_ids": []}), valid) == ([], 0, None)


def test_parse_doc_ids_failures_mean_no_routing():
    valid = {"IF01"}
    assert dr.parse_doc_ids("not json", valid)[0] == [] and dr.parse_doc_ids("not json", valid)[2]
    assert dr.parse_doc_ids(json.dumps({"doc_ids": "IF01"}), valid) == ([], 0, "doc_ids가 문자열 목록이 아님")


def test_route_records_errors_instead_of_raising():
    class Boom:
        def invoke(self, _):
            raise RuntimeError("boom")

    class Ok:
        def invoke(self, _):
            return SimpleNamespace(content=json.dumps({"doc_ids": ["IF01", "IF77"]}), usage_metadata={"input_tokens": 7, "output_tokens": 3}, response_metadata={})

    spec = dr.load_prompt()
    assert "RuntimeError" in dr.route(Boom(), spec, "IF01 | t", {"IF01"}, "q")["error"]
    good = dr.route(Ok(), spec, "IF01 | t", {"IF01"}, "q")
    assert good["doc_ids"] == ["IF01"] and good["invalid_ids"] == 1 and good["usage"]["input_tokens"] == 7


# ---- 문서 한정 검색 ----
def _chunk(cid, book, page):
    return {"chunk_id": cid, "book_id": book, "file_name": f"{book}.pdf", "page": page, "chunk_text": f"text-{cid}", "metadata": {}, "chunk_index": 1,
            "chunking_version": "v", "dataset_version": "d"}


class _FakeEmbedder:
    def embed_queries(self, texts):
        return [[1.0, 0.0]]


class _FakeBM25:
    def __init__(self, scores):
        self.scores = scores

    def get_scores(self, tokens):
        return np.array(self.scores)


def _fake_hybrid():
    chunks = [_chunk("a1", "IF01", 1), _chunk("a2", "IF01", 2), _chunk("b1", "IF02", 1), _chunk("b2", "IF02", 2)]
    vectors = np.array([[1, 0], [0.9, 0.436], [0.2, 0.98], [0.1, 0.995]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = NumpyIndex(vectors, chunks, {})
    dense = SimpleNamespace(embedder=_FakeEmbedder(), index=index)
    sparse = SimpleNamespace(chunks=chunks, bm25=_FakeBM25([5.0, 4.0, 1.0, 0.5]), tokenize=lambda q: q.split())

    class H:
        config = HybridConfig(final_k=5, fused_top=20)

        def __init__(self):
            self.dense, self.sparse = dense, sparse
            self.calls = []

        def describe(self):
            return {"fusion": "rrf"}

        def retrieve(self, question, k=None):
            self.calls.append(k)
            top = [{"rank": 1, "score": 1.0, "chunk_id": "a1", "book_id": "IF01", "file_name": "IF01.pdf", "page": 1, "text": "text-a1", "metadata": {}}]
            return RetrievalResult(question=question, method="hybrid", k=k, chunks=top)

    return H()


def test_doc_routed_hybrid_adds_chunks_from_routed_documents_only_in_the_restricted_list():
    h = _fake_hybrid()
    routed = DocRoutedHybrid(h, {"Q": ["IF02"]}, top=20)
    res = routed.retrieve("Q", k=5)
    ids = [c["chunk_id"] for c in res.chunks]
    assert ids[0] in ("a1", "b1") and set(ids) == {"a1", "b1", "b2"}  # 전체 후보 a1 + IF02 문서 안의 후보 b1·b2
    assert res.candidates["routed_docs"] == ["IF02"] and routed.describe()["doc_routed"] is True and [c["rank"] for c in res.chunks] == [1, 2, 3]


def test_doc_routed_hybrid_without_routing_equals_the_plain_candidates():
    h = _fake_hybrid()
    res = DocRoutedHybrid(h, {}, top=20).retrieve("Q", k=5)
    assert [c["chunk_id"] for c in res.chunks] == ["a1"] and res.candidates["routed_docs"] == []
