"""PDF 추출 (STEP 1).

PDF 1개를 읽어 `documents.jsonl`의 1행(Document 매니페스트)과 `pages.jsonl`의 행들(Extracted Page)을 만든다.
필드 정의는 project_docs/02_implementation/09_DATA_SCHEMA.md §6.1, §6.2를 따른다.

원칙
- 추출한 텍스트는 가공하지 않고 그대로 저장한다 (머리말·꼬리말·줄바꿈 포함). 줄 잇기와 머리말·꼬리말 처리는 청킹 단계에서 한다.
- 페이지 번호는 PDF의 실제 위치이며 첫 페이지가 1이다.
- 문서·페이지 단위 실패는 예외로 중단하지 않고 상태와 사유를 기록한다 (빠진 문서와 처리 실패를 기록).
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from pypdf import PdfReader

EXTRACTION_METHOD = "pypdf"


def extraction_version() -> str:
    """추출기와 설정의 버전 문자열. 설정은 pypdf 기본 `extract_text()`를 그대로 쓴다."""
    return f"pypdf-{version('pypdf')}/default"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_posix(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def extract_pdf(
    pdf_path: Path,
    *,
    repo_root: Path,
    dataset_id: str,
    dataset_version: str,
) -> tuple[dict, list[dict]]:
    """PDF 1개를 처리해 (문서 레코드, 페이지 레코드 목록)을 돌려준다. 예외를 밖으로 던지지 않는다."""
    started = time.perf_counter()
    book_id = pdf_path.stem  # 파일명에서 .pdf를 뺀 값
    doc: dict = {
        "book_id": book_id,
        "file_name": pdf_path.name,
        "file_path": _relative_posix(pdf_path, repo_root),
        "file_sha256": None,
        "file_size": None,
        "page_count": None,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "extraction_status": "failed",
        "extraction_seconds": None,
        "error_message": None,
        "created_at": now_utc(),
    }
    pages: list[dict] = []
    ext_version = extraction_version()

    try:
        doc["file_size"] = pdf_path.stat().st_size
        doc["file_sha256"] = sha256_file(pdf_path)
    except OSError as exc:
        doc["error_message"] = f"file_read_error: {type(exc).__name__}"
        doc["extraction_seconds"] = round(time.perf_counter() - started, 3)
        return doc, pages

    try:
        reader = PdfReader(str(pdf_path))
        if reader.is_encrypted:
            raise ValueError("encrypted_pdf")
        page_total = len(reader.pages)
    except Exception as exc:  # noqa: BLE001 - 손상·암호화 등 어떤 오류든 기록하고 계속한다
        doc["error_message"] = f"open_error: {type(exc).__name__}: {str(exc)[:120]}"
        doc["extraction_seconds"] = round(time.perf_counter() - started, 3)
        return doc, pages

    doc["page_count"] = page_total
    for index, page in enumerate(reader.pages, start=1):
        record = {
            "book_id": book_id,
            "page": index,
            "text": "",
            "extraction_method": EXTRACTION_METHOD,
            "extraction_version": ext_version,
            "extraction_status": "success",
            "error_message": None,
            "extracted_at": now_utc(),
        }
        try:
            text = page.extract_text() or ""
            record["text"] = text
            if not text.strip():
                record["extraction_status"] = "partial"
                record["error_message"] = "no_text_extracted"
        except Exception as exc:  # noqa: BLE001
            record["extraction_status"] = "failed"
            record["error_message"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        pages.append(record)

    statuses = {p["extraction_status"] for p in pages}
    if page_total == 0 or statuses == {"failed"}:
        doc["extraction_status"] = "failed"
        doc["error_message"] = "no_pages" if page_total == 0 else "all_pages_failed"
    elif statuses == {"success"}:
        doc["extraction_status"] = "success"
    else:
        doc["extraction_status"] = "partial"
        bad = sum(1 for p in pages if p["extraction_status"] != "success")
        doc["error_message"] = f"{bad}_of_{page_total}_pages_not_success"

    doc["extraction_seconds"] = round(time.perf_counter() - started, 3)
    return doc, pages


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
