"""numpy 정확 검색 저장소 (확장 브랜치, D-047). 모델·네트워크 없이 확인한다."""
import numpy as np
import pytest

from src.index.numpy_store import NumpyIndex


def _chunk(i):
    return {"chunk_id": f"c{i}", "book_id": "B", "file_name": "B.pdf", "page": i + 1, "chunk_index": 0, "chunk_text": f"t{i}",
            "metadata": {}, "chunking_version": "chunk-v1", "dataset_version": "d-v1"}


def _unit(rows):
    a = np.asarray(rows, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def _index():
    v = _unit([[1, 0, 0], [0, 1, 0], [0.8, 0.6, 0], [0, 0, 1]])
    return NumpyIndex.build([_chunk(i) for i in range(4)], v, {"name": "t"})


def test_search_returns_top_k_by_cosine_with_payload_fields():
    hits = _index().search([1, 0, 0], k=2)
    assert [h["chunk_id"] for h in hits] == ["c0", "c2"]
    assert hits[0]["score"] == pytest.approx(1.0) and hits[1]["score"] == pytest.approx(0.8)
    assert {"chunk_text", "book_id", "file_name", "page", "metadata"} <= set(hits[0])


def test_ties_are_broken_by_chunk_order_so_results_are_stable():
    v = _unit([[1, 0], [1, 0], [1, 0]])
    idx = NumpyIndex.build([_chunk(i) for i in range(3)], v, {})
    assert [h["chunk_id"] for h in idx.search([1, 0], k=3)] == ["c0", "c1", "c2"]


def test_build_rejects_unnormalized_vectors_and_wrong_query_dimension():
    with pytest.raises(ValueError, match="정규화"):
        NumpyIndex.build([_chunk(0)], [[2.0, 0.0, 0.0]], {})
    with pytest.raises(ValueError, match="차원"):
        _index().search([1, 0], k=1)


def test_payload_does_not_carry_evaluation_fields():
    c = _chunk(0) | {"reference_answer": "정답", "expected_action": "answer"}
    idx = NumpyIndex.build([c], _unit([[1, 0, 0]]), {})
    assert "reference_answer" not in idx.payloads[0] and "expected_action" not in idx.payloads[0]


def test_save_load_round_trip_and_never_overwrites(tmp_path):
    idx = _index()
    idx.save("x", base=tmp_path)
    again = NumpyIndex.load("x", base=tmp_path)
    assert len(again) == 4 and again.meta["count"] == 4 and again.meta["dimension"] == 3
    assert [h["chunk_id"] for h in again.search([0, 0, 1], k=1)] == ["c3"]
    with pytest.raises(FileExistsError):
        idx.save("x", base=tmp_path)
