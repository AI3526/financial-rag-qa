"""STEP 2: data/processed/pages.jsonl을 청킹해 chunks.jsonl과 chunking_manifest.json을 만든다.

사용: uv run python scripts/chunk_pages.py [--dataset basic|intermediate|advanced] [--force] [--no-langfuse]

- 입력은 STEP 1 결과(documents.jsonl, pages.jsonl)뿐이다 (evaluation/ 은 읽지 않는다).
- 기존 결과를 덮어쓰지 않는다. 이미 있으면 --force 없이는 중단한다 (CLAUDE.md §4).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingest import chunk as ck  # noqa: E402
from src.ingest.extract import read_jsonl, write_jsonl  # noqa: E402
from src.observability.ingestion_trace import ChunkingTracer  # noqa: E402
from src.observability.langfuse_client import get_langfuse  # noqa: E402

OUT_DIRS = {"basic": ROOT / "data" / "processed", "intermediate": ROOT / "data" / "processed_intermediate",
            "advanced": ROOT / "data" / "processed_advanced"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="basic", choices=sorted(OUT_DIRS), help="청킹할 데이터셋(기본 basic)")
    parser.add_argument("--force", action="store_true", help="기존 chunks.jsonl/chunking_manifest.json을 덮어쓴다")
    parser.add_argument("--no-langfuse", action="store_true", help="Langfuse 기록을 건너뛴다 (오프라인 실행)")
    args = parser.parse_args()
    OUT_DIR = OUT_DIRS[args.dataset]

    docs_path, pages_path = OUT_DIR / "documents.jsonl", OUT_DIR / "pages.jsonl"
    chunks_path, manifest_path = OUT_DIR / "chunks.jsonl", OUT_DIR / "chunking_manifest.json"
    if not (docs_path.exists() and pages_path.exists()):
        print("STEP 1 결과(documents.jsonl, pages.jsonl)가 없습니다. 먼저 ingest_pdfs.py를 실행하세요.")
        return 1
    if (chunks_path.exists() or manifest_path.exists()) and not args.force:
        print("이미 결과 파일이 있습니다. 덮어쓰려면 --force를 사용하세요. (중단)")
        return 1

    docs, pages = read_jsonl(docs_path), read_jsonl(pages_path)
    params = ck.ChunkParams()
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    langfuse = None if args.no_langfuse else get_langfuse()
    meta = {"step": "2", "chunking_version": ck.CHUNKING_VERSION, "chunking_method": ck.CHUNKING_METHOD,
            "dataset_version": docs[0]["dataset_version"], "doc_count": len(docs)}

    all_chunks: list[dict] = []
    all_warnings: list[dict] = []
    print(f"청킹: {ck.CHUNKING_METHOD}/{ck.CHUNKING_VERSION} params={params.as_dict()}, 문서 {len(docs)}개")
    with ChunkingTracer(langfuse, meta) as tracer:
        for doc in docs:
            doc_pages = [p for p in pages if p["book_id"] == doc["book_id"]]
            chunks, warnings = tracer.chunk(
                doc,
                lambda d=doc, dp=doc_pages: ck.chunk_document(d, dp, params, created_at=created_at),
                params.as_dict(),
            )
            all_chunks += chunks
            all_warnings += warnings
            print(f"  {doc['book_id']}: chunks={len(chunks)} warnings={len(warnings)}")
    if langfuse is None:
        print("Langfuse: 기록하지 않음 (--no-langfuse 또는 설정 없음)")
    else:
        print(f"Langfuse: 문서 Span {tracer.documents_recorded}개 기록, trace_id={tracer.trace_id}")
        print(f"  {tracer.trace_url()}")

    sizes = [len(c["chunk_text"]) for c in all_chunks]
    stats = {
        "chunk_count": len(all_chunks),
        "avg_chars": round(statistics.mean(sizes), 1) if sizes else None,
        "median_chars": statistics.median(sizes) if sizes else None,
        "min_chars": min(sizes) if sizes else None,
        "max_chars": max(sizes) if sizes else None,
        "chunks_per_book": {d["book_id"]: sum(1 for c in all_chunks if c["book_id"] == d["book_id"]) for d in docs},
    }
    manifest = {
        "chunking_method": ck.CHUNKING_METHOD,
        "chunking_version": ck.CHUNKING_VERSION,
        "params": params.as_dict(),
        "dataset_id": docs[0]["dataset_id"],
        "dataset_version": docs[0]["dataset_version"],
        "input": {"documents": len(docs), "pages": len(pages)},
        "stats": stats,
        "warnings": all_warnings,
        "created_at": created_at,
    }
    write_jsonl(chunks_path, all_chunks)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"청크 {stats['chunk_count']}개 저장 -> {chunks_path}")
    print(f"글자 수: 평균 {stats['avg_chars']}, 중앙 {stats['median_chars']}, 최소 {stats['min_chars']}, 최대 {stats['max_chars']}")
    print(f"경고: {len(all_warnings)}건")
    return 0 if not all_warnings else 2


if __name__ == "__main__":
    sys.exit(main())
