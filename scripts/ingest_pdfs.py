"""STEP 1: documents/*.pdf 전체를 추출해 data/processed/에 저장한다.

사용: uv run python scripts/ingest_pdfs.py [--force]

- 검색 대상은 documents/*.pdf뿐이다 (evaluation/ 은 읽지 않는다).
- 기존 결과를 덮어쓰지 않는다. 이미 있으면 --force 없이는 중단한다 (CLAUDE.md §4).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingest.extract import extraction_version, write_jsonl  # noqa: E402
from src.observability.ingestion_trace import IngestionTracer  # noqa: E402
from src.observability.langfuse_client import get_langfuse  # noqa: E402

DATASET_ID = "basic"
DATASET_VERSION = "basic-v1"
DOCS_DIR = ROOT / "data" / "finance_basic" / "documents"
OUT_DIR = ROOT / "data" / "processed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="기존 documents.jsonl/pages.jsonl을 덮어쓴다")
    parser.add_argument("--no-langfuse", action="store_true", help="Langfuse 기록을 건너뛴다 (오프라인 실행)")
    args = parser.parse_args()

    docs_path = OUT_DIR / "documents.jsonl"
    pages_path = OUT_DIR / "pages.jsonl"
    if (docs_path.exists() or pages_path.exists()) and not args.force:
        print("이미 결과 파일이 있습니다. 덮어쓰려면 --force를 사용하세요. (중단)")
        return 1

    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"PDF가 없습니다: {DOCS_DIR}")
        return 1

    print(f"추출기: {extraction_version()}, 대상 PDF {len(pdfs)}개")
    all_docs: list[dict] = []
    all_pages: list[dict] = []
    langfuse = None if args.no_langfuse else get_langfuse()
    meta = {"step": "1", "extraction_version": extraction_version(),
            "dataset_version": DATASET_VERSION, "pdf_count": len(pdfs)}
    with IngestionTracer(langfuse, meta) as tracer:
        for pdf in pdfs:
            doc, pages = tracer.extract(
                pdf, repo_root=ROOT, dataset_id=DATASET_ID, dataset_version=DATASET_VERSION
            )
            all_docs.append(doc)
            all_pages.extend(pages)
            print(f"  {doc['book_id']}: {doc['extraction_status']:7} pages={doc['page_count']} "
                  f"{doc['extraction_seconds']}s sha256={str(doc['file_sha256'])[:10]} {doc['error_message'] or ''}")
    if langfuse is None:
        print("Langfuse: 기록하지 않음 (--no-langfuse 또는 설정 없음)")
    else:
        print(f"Langfuse: 문서 Span {tracer.documents_recorded}개 기록, trace_id={tracer.trace_id}")
        print(f"  {tracer.trace_url()}")

    write_jsonl(docs_path, all_docs)
    write_jsonl(pages_path, all_pages)

    failed = [d["book_id"] for d in all_docs if d["extraction_status"] != "success"]
    print(f"문서 {len(all_docs)}개, 페이지 {len(all_pages)}개 저장 -> {OUT_DIR}")
    print(f"success 아닌 문서: {failed if failed else '없음'}")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
