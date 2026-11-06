from contextlib import contextmanager

from pypdf import PdfWriter

from src.observability.ingestion_trace import IngestionTracer

KW = dict(dataset_id="basic", dataset_version="basic-v1")


def _pdf(path, pages=2):
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    with path.open("wb") as fh:
        writer.write(fh)


class _Span:
    def __init__(self, log, name, kwargs):
        self.log, self.name, self.kwargs = log, name, kwargs
        self.updates = []

    def update(self, **kw):
        self.updates.append(kw)


class StubLangfuse:
    """Langfuse 클라이언트 흉내. 호출을 기록만 한다."""

    def __init__(self):
        self.spans = []
        self.flushed = False

    def create_trace_id(self, seed=None):
        return "a" * 32

    @contextmanager
    def start_as_current_observation(self, **kwargs):
        span = _Span(self, kwargs.get("name"), kwargs)
        self.spans.append(span)
        yield span

    def flush(self):
        self.flushed = True

    def get_trace_url(self, trace_id=None):
        return f"https://example.invalid/traces/{trace_id}"


class BrokenLangfuse(StubLangfuse):
    def start_as_current_observation(self, **kwargs):
        raise RuntimeError("langfuse down")

    def flush(self):
        raise RuntimeError("flush failed")


def test_works_without_langfuse(tmp_path):
    pdf = tmp_path / "A01.pdf"
    _pdf(pdf)
    with IngestionTracer(None) as tracer:
        doc, pages = tracer.extract(pdf, repo_root=tmp_path, **KW)
    assert doc["page_count"] == 2 and len(pages) == 2
    assert tracer.trace_id is None and tracer.trace_url() is None


def test_records_one_trace_and_one_span_per_pdf_with_text_in_output(tmp_path):
    pdfs = []
    for name in ("A01", "A02"):
        p = tmp_path / f"{name}.pdf"
        _pdf(p)
        pdfs.append(p)
    lf = StubLangfuse()
    with IngestionTracer(lf, {"step": "1"}) as tracer:
        for p in pdfs:
            tracer.extract(p, repo_root=tmp_path, **KW)
    names = [s.name for s in lf.spans]
    # Trace 1 + Span 2. Span 이름에 원본 PDF 파일명이 반영된다
    assert names == ["document-ingestion", "extract-pdf A01.pdf", "extract-pdf A02.pdf"]
    assert lf.spans[1].kwargs["metadata"] == {"book_id": "A01"}
    assert tracer.documents_recorded == 2
    assert lf.flushed
    out = lf.spans[1].updates[0]["output"]
    assert out["summary"]["book_id"] == "A01"
    assert [p["page"] for p in out["pages"]] == [1, 2]  # 페이지별 텍스트(여기서는 빈 문서라 빈 문자열)
    assert "text" in out["pages"][0]
    assert tracer.trace_id == "a" * 32
    assert tracer.trace_url().endswith("a" * 32)


def test_failed_pdf_is_marked_as_error_level(tmp_path):
    bad = tmp_path / "BAD01.pdf"
    bad.write_text("pdf 아님", encoding="utf-8")
    lf = StubLangfuse()
    with IngestionTracer(lf) as tracer:
        doc, _ = tracer.extract(bad, repo_root=tmp_path, **KW)
    assert doc["extraction_status"] == "failed"
    update = lf.spans[1].updates[0]
    assert update["level"] == "ERROR"
    assert update["status_message"].startswith("open_error")


def test_environment_is_passed_to_langfuse_and_matches_track(tmp_path, monkeypatch):
    import langfuse

    captured = {}

    @contextmanager
    def fake_propagate(**kwargs):
        captured.update(kwargs)
        yield

    monkeypatch.setattr(langfuse, "propagate_attributes", fake_propagate)
    pdf = tmp_path / "A01.pdf"
    _pdf(pdf)
    with IngestionTracer(StubLangfuse(), {"step": "1"}, environment="development") as tracer:
        tracer.extract(pdf, repo_root=tmp_path, **KW)
    assert captured["environment"] == "development"
    assert captured["trace_name"] == "document-ingestion"
    assert captured["tags"] == ["ingestion"]
    assert captured["metadata"]["track"] == "development"  # environment와 같은 값
    assert captured["metadata"]["step"] == "1"


def test_invalid_environment_is_rejected():
    for bad in ("Development", "langfuse-x", "dev env", "", "공식"):
        try:
            IngestionTracer(None, environment=bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} 은 거부되어야 한다")
    IngestionTracer(None, environment="official")  # 올바른 값은 통과


def test_langfuse_failure_never_blocks_extraction(tmp_path):
    pdf = tmp_path / "A01.pdf"
    _pdf(pdf)
    with IngestionTracer(BrokenLangfuse()) as tracer:
        doc, pages = tracer.extract(pdf, repo_root=tmp_path, **KW)  # 예외 없이 끝나야 한다
    assert doc["page_count"] == 2 and len(pages) == 2
    assert tracer.documents_recorded == 0  # 기록은 못 했지만 추출 결과는 그대로
