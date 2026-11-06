import json

from pypdf import PdfWriter

from src.ingest import extract as ex


def _blank_pdf(path, pages=2):
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    with path.open("wb") as fh:
        writer.write(fh)


def test_sha256_is_stable_and_matches_content(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"abc")
    # sha256("abc")
    assert ex.sha256_file(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_non_pdf_file_is_recorded_as_failed_not_raised(tmp_path):
    bad = tmp_path / "BAD01.pdf"
    bad.write_text("이것은 PDF가 아닙니다", encoding="utf-8")
    doc, pages = ex.extract_pdf(bad, repo_root=tmp_path, dataset_id="basic", dataset_version="basic-v1")
    assert doc["book_id"] == "BAD01"
    assert doc["extraction_status"] == "failed"
    assert doc["error_message"] and doc["error_message"].startswith("open_error")
    assert doc["file_sha256"]  # 읽을 수 있었으므로 해시는 기록된다
    assert doc["page_count"] is None
    assert pages == []


def test_missing_file_is_recorded_as_failed(tmp_path):
    doc, pages = ex.extract_pdf(tmp_path / "NONE.pdf", repo_root=tmp_path, dataset_id="basic", dataset_version="basic-v1")
    assert doc["extraction_status"] == "failed"
    assert doc["error_message"].startswith("file_read_error")
    assert doc["file_sha256"] is None  # 확인하지 못한 값은 0이나 빈 문자열이 아니라 null
    assert pages == []


def test_blank_pages_are_partial_with_reason(tmp_path):
    pdf = tmp_path / "BLK01.pdf"
    _blank_pdf(pdf, pages=2)
    doc, pages = ex.extract_pdf(pdf, repo_root=tmp_path, dataset_id="basic", dataset_version="basic-v1")
    assert doc["page_count"] == 2
    assert [p["page"] for p in pages] == [1, 2]  # 첫 페이지가 1
    assert all(p["extraction_status"] == "partial" and p["error_message"] == "no_text_extracted" for p in pages)
    assert doc["extraction_status"] == "partial"
    assert doc["file_path"] == "BLK01.pdf"  # 저장소 기준 상대 경로


def test_record_fields_follow_schema(tmp_path):
    pdf = tmp_path / "SCH01.pdf"
    _blank_pdf(pdf, pages=1)
    doc, pages = ex.extract_pdf(pdf, repo_root=tmp_path, dataset_id="basic", dataset_version="basic-v1")
    assert set(doc) == {
        "book_id", "file_name", "file_path", "file_sha256", "file_size", "page_count",
        "dataset_id", "dataset_version", "extraction_status", "extraction_seconds",
        "error_message", "created_at",
    }
    assert set(pages[0]) == {
        "book_id", "page", "text", "extraction_method", "extraction_version",
        "extraction_status", "error_message", "extracted_at",
    }
    assert pages[0]["extraction_method"] == "pypdf"
    assert pages[0]["extraction_version"].startswith("pypdf-")


def test_jsonl_roundtrip_keeps_korean_and_newlines(tmp_path):
    rows = [{"book_id": "FB01", "page": 1, "text": "해온솔루션\n운\n영서비스"}]
    path = tmp_path / "out" / "pages.jsonl"
    ex.write_jsonl(path, rows)
    raw = path.read_text(encoding="utf-8")
    assert "해온솔루션" in raw  # ensure_ascii=False
    assert json.loads(raw.splitlines()[0]) == rows[0]
    assert ex.read_jsonl(path) == rows
