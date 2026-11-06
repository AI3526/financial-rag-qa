from contextlib import contextmanager

from src.ingest import chunk as ck
from src.observability.ingestion_trace import ChunkingTracer
from tests.unit.test_ingestion_trace import BrokenLangfuse, StubLangfuse

DOC = {"book_id": "FB01", "file_name": "FB01.pdf", "dataset_id": "basic", "dataset_version": "basic-v1"}
PAGE = {"book_id": "FB01", "page": 1, "extraction_status": "success",
        "text": "FB01 | 해온솔루션 경영기획실\n문서 제목\n소제목\n" + "가" * 40 + "\n끝\n1 / 24"}


def _compute():
    return ck.chunk_document(DOC, [PAGE], ck.ChunkParams(), created_at="t")


def test_records_trace_and_one_span_per_document_with_chunks_in_output():
    lf = StubLangfuse()
    with ChunkingTracer(lf, {"step": "2"}) as tracer:
        chunks, warnings = tracer.chunk(DOC, _compute, ck.ChunkParams().as_dict())
    assert [s.name for s in lf.spans] == ["document-chunking", "chunk-pdf FB01.pdf"]  # 파일명이 이름에 반영
    out = lf.spans[1].updates[0]["output"]
    assert out["summary"]["chunk_count"] == len(chunks) == 1
    assert out["chunks"][0]["chunk_id"] == "FB01-p1-c1"
    assert out["chunks"][0]["text"] == chunks[0]["chunk_text"]  # 청크 텍스트가 기록된다
    assert lf.spans[1].kwargs["input"]["params"]["max_chars"] == 450
    assert lf.spans[1].kwargs["metadata"] == {"book_id": "FB01"}
    assert tracer.documents_recorded == 1 and lf.flushed


def test_trace_uses_chunking_name_tags_and_environment(monkeypatch):
    import langfuse

    captured = {}

    @contextmanager
    def fake_propagate(**kwargs):
        captured.update(kwargs)
        yield

    monkeypatch.setattr(langfuse, "propagate_attributes", fake_propagate)
    with ChunkingTracer(StubLangfuse(), environment="development"):
        pass
    assert captured["trace_name"] == "document-chunking"
    assert captured["tags"] == ["chunking"]
    assert captured["environment"] == "development"


def test_document_without_chunks_is_marked_error():
    lf = StubLangfuse()
    with ChunkingTracer(lf) as tracer:
        tracer.chunk(DOC, lambda: ([], [{"book_id": "FB01", "page": 1, "reason": "no_text"}]), {})
    update = lf.spans[1].updates[0]
    assert update["level"] == "ERROR" and update["status_message"] == "no_chunk"


def test_works_without_langfuse_and_langfuse_failure_never_blocks():
    with ChunkingTracer(None) as tracer:
        chunks, _ = tracer.chunk(DOC, _compute, {})
    assert len(chunks) == 1
    with ChunkingTracer(BrokenLangfuse()) as tracer:
        chunks, _ = tracer.chunk(DOC, _compute, {})  # 예외 없이 끝나야 한다
    assert len(chunks) == 1 and tracer.documents_recorded == 0
