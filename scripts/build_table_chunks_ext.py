"""확장 브랜치 3단계: chunk-v1 청크에 '표 행 문장 청크'를 추가한 청크 집합(chunk-v3a)을 만든다 (EXTENSION_INTERMEDIATE_LOG).

사용 (pdfplumber는 프로젝트에 설치하지 않고 임시 환경으로만 쓴다):
    uv run --with pdfplumber python scripts/build_table_chunks_ext.py --dataset intermediate

- 입력: `data/processed_{dataset}/`의 chunks.jsonl·documents.jsonl·pages.jsonl과 원본 PDF(표 구조를 읽는 용도). 평가 파일은 읽지 않는다.
- 출력: `data/processed_{dataset}_v3a/`(chunks.jsonl, chunking_manifest.json). 기존 chunk-v1 결과는 건드리지 않는다. 이미 있으면 중단한다.
- 변경은 하나뿐이다: chunk-v1 청크는 그대로 두고, 쪽마다 표를 행 문장으로 바꾼 보조 청크(`{book}-p{page}-t{n}`)를 **추가**한다.
  보조 청크는 쪽과 같은 머리말·문서 제목·소제목 metadata를 가져 검색 입력(`retrieval_text`) 앞의 맥락이 원래 청크와 같다.
- pdfplumber 0.11.10(MIT, 패키지 메타데이터로 확인)을 쓴다. 표는 선이 있는 표로 쪽마다 1개씩 있고 첫 행이 머리줄이다(전수 점검으로 확인).
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
from src.ingest import table_rows as tr  # noqa: E402
from src.ingest.extract import read_jsonl, write_jsonl  # noqa: E402

DATASETS = {"intermediate": {"processed": ROOT / "data" / "processed_intermediate", "docs": ROOT / "data" / "finance_intermediate" / "documents"}}
VERSION = "chunk-v3a-tablerows"
METHOD = "table-row-sentences"
PACK_MAX_CHARS = 500


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="intermediate", choices=sorted(DATASETS))
    args = ap.parse_args()
    try:
        import pdfplumber
        from importlib.metadata import version as pkg_version
    except ImportError:
        print("pdfplumber가 필요합니다: uv run --with pdfplumber python scripts/build_table_chunks_ext.py (프로젝트에는 설치하지 않음)")
        return 1
    d = DATASETS[args.dataset]
    out_dir = d["processed"].parent / f"{d['processed'].name}_v3a"
    if (out_dir / "chunks.jsonl").exists():
        print(f"이미 결과가 있습니다(덮어쓰지 않음): {out_dir}. (중단)")
        return 1

    docs = read_jsonl(d["processed"] / "documents.jsonl")
    pages = {(p["book_id"], p["page"]): p for p in read_jsonl(d["processed"] / "pages.jsonl")}
    base_chunks = read_jsonl(d["processed"] / "chunks.jsonl")
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    index = max(c["chunk_index"] for c in base_chunks)
    extra: list[dict] = []
    stats = {"pages": 0, "tables": 0, "tables_header_like": 0, "rows": 0, "empty_tables": 0}
    for doc in docs:
        with pdfplumber.open(d["docs"] / doc["file_name"]) as pdf:
            for number, page in enumerate(pdf.pages, start=1):
                stats["pages"] += 1
                meta, _, _ = ck.parse_page(pages[(doc["book_id"], number)]["text"])
                for t_no, table in enumerate(page.find_tables(), start=1):
                    rows = table.extract()
                    stats["tables"] += 1
                    cells = [tr.clean_cell(c) for c in (rows[0] if rows else [])]
                    stats["tables_header_like"] += tr.is_header_like(cells)
                    sentences = tr.serialize_table(rows)
                    if not sentences:
                        stats["empty_tables"] += 1
                        continue
                    stats["rows"] += len(sentences)
                    for part, text in enumerate(tr.pack_sentences(sentences, PACK_MAX_CHARS), start=1):
                        index += 1
                        suffix = f"t{t_no}" if part == 1 else f"t{t_no}-{part}"
                        extra.append({
                            "chunk_id": f"{doc['book_id']}-p{number}-{suffix}", "book_id": doc["book_id"], "file_name": doc["file_name"], "page": number,
                            "chunk_text": text, "chunk_index": index, "chunking_method": METHOD, "chunking_version": VERSION,
                            "dataset_id": doc["dataset_id"], "dataset_version": doc["dataset_version"], "token_count": None,
                            "metadata": {**meta, "derived": "table-rows", "table_index": t_no}, "created_at": created_at})
    all_chunks = [{**c, "chunking_version": VERSION} for c in base_chunks] + extra
    ids = [c["chunk_id"] for c in all_chunks]
    assert len(ids) == len(set(ids)), "chunk_id가 겹칩니다"
    out_dir.mkdir(parents=True)
    write_jsonl(out_dir / "chunks.jsonl", all_chunks)
    sizes = [len(c["chunk_text"]) for c in extra]
    manifest = {"chunking_method": f"{ck.CHUNKING_METHOD}+{METHOD}", "chunking_version": VERSION, "base_chunking_version": ck.CHUNKING_VERSION,
                "params": {"base": ck.ChunkParams().as_dict(), "table_sentence_pack_max_chars": PACK_MAX_CHARS},
                "dataset_id": docs[0]["dataset_id"], "dataset_version": docs[0]["dataset_version"], "tool": {"pdfplumber": pkg_version("pdfplumber")},
                "input": {"documents": len(docs), "pages": stats["pages"]},
                "stats": {"chunk_count": len(all_chunks), "base_chunks": len(base_chunks), "table_chunks": len(extra), "tables": stats["tables"],
                          "tables_header_like": stats["tables_header_like"], "tables_without_rows": stats["empty_tables"], "table_rows": stats["rows"],
                          "table_chunk_avg_chars": round(statistics.mean(sizes), 1), "table_chunk_max_chars": max(sizes)},
                "created_at": created_at}
    (out_dir / "chunking_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"표 {stats['tables']}개 (머리줄로 보이는 첫 행 {stats['tables_header_like']}개, 데이터 행 없음 {stats['empty_tables']}개), 행 문장 {stats['rows']}개")
    print(f"표 청크 {len(extra)}개 추가(평균 {manifest['stats']['table_chunk_avg_chars']}자, 최대 {manifest['stats']['table_chunk_max_chars']}자) → 전체 {len(all_chunks)}개, {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
