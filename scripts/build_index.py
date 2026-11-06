"""STEP 3: chunks.jsonl을 임베딩해 Qdrant 컬렉션을 만들고 색인 매니페스트를 남긴다.

사용: uv run python scripts/build_index.py --model e5|bgem3 [--no-langfuse]

- 컬렉션 이름: basic__chunk-v1__{임베딩버전}. 이미 있으면 덮어쓰지 않고 중단한다 (삭제도 하지 않는다).
- 입력은 STEP 1·2 결과뿐이다 (evaluation/ 은 읽지 않는다).
- 결과: data/processed/index_manifest__{임베딩버전}.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_env  # noqa: E402

load_env()  # huggingface_hub보다 먼저: 빈 HF_HOME 등을 제거

from src.index import qdrant_store as qs  # noqa: E402
from src.index.embedding import CANDIDATES, Embedder, collection_name  # noqa: E402
from src.ingest import chunk as ck  # noqa: E402
from src.ingest.extract import read_jsonl  # noqa: E402
from src.observability.ingestion_trace import IndexingTracer  # noqa: E402
from src.observability.langfuse_client import get_langfuse  # noqa: E402

OUT_DIR = ROOT / "data" / "processed"
BATCH = 64


def model_license(hf_id: str) -> str | None:
    """모델 카드에서 확인한 라이선스. 확인하지 못하면 None (추측하지 않는다)."""
    try:
        from huggingface_hub import model_info

        return model_info(hf_id).card_data.license
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(CANDIDATES))
    parser.add_argument("--no-langfuse", action="store_true")
    args = parser.parse_args()
    spec = CANDIDATES[args.model]

    docs = read_jsonl(OUT_DIR / "documents.jsonl")
    chunks = read_jsonl(OUT_DIR / "chunks.jsonl")
    chunking = json.loads((OUT_DIR / "chunking_manifest.json").read_text(encoding="utf-8"))
    dataset_id, chunking_version = chunking["dataset_id"], chunking["chunking_version"]
    name = collection_name(dataset_id, chunking_version, spec)
    manifest_path = OUT_DIR / f"index_manifest__{spec.version_tag}.json"

    client = qs.get_client()
    mode = qs.qdrant_mode()
    if qs.collection_exists(client, name) or manifest_path.exists():
        print(f"이미 있습니다(덮어쓰지 않음): 컬렉션 {name} 또는 {manifest_path.name}. (중단)")
        return 1

    print(f"Qdrant: {mode} 모드, 컬렉션 {name}")
    t0 = time.perf_counter()
    embedder = Embedder(spec)
    model_load = time.perf_counter() - t0
    print(f"모델 로드 {model_load:.1f}s device={embedder.device} dim={embedder.dimension} revision={embedder.revision}")

    texts = {c["chunk_id"]: ck.retrieval_text(c) for c in chunks}
    tokens = embedder.token_lengths(list(texts.values()))
    over = sum(1 for n in tokens if n > spec.max_seq_length)
    print(f"토큰 수(접두어 포함): 평균 {statistics.mean(tokens):.0f}, 최대 {max(tokens)}, 한도 {spec.max_seq_length} 초과 {over}개")

    qs.create_collection(client, name, embedder.dimension)
    langfuse = None if args.no_langfuse else get_langfuse()
    meta = {"step": "3", "embedding_model": spec.hf_id, "embedding_revision": str(embedder.revision),
            "collection": name, "chunking_version": chunking_version, "doc_count": len(docs)}
    embed_total = upsert_total = 0.0
    failed: list[dict] = []
    per_doc: list[dict] = []
    t_all = time.perf_counter()
    with IndexingTracer(langfuse, meta) as tracer:
        for doc in docs:
            doc_chunks = [c for c in chunks if c["book_id"] == doc["book_id"]]

            def work(dc=doc_chunks, d=doc) -> dict:
                result = {"book_id": d["book_id"], "chunks": len(dc), "upserted": 0}
                try:
                    t = time.perf_counter()
                    vectors: list[list[float]] = []
                    for i in range(0, len(dc), BATCH):
                        vectors += embedder.embed_passages([texts[c["chunk_id"]] for c in dc[i:i + BATCH]])
                    result["embed_seconds"] = round(time.perf_counter() - t, 3)
                    t = time.perf_counter()
                    qs.upsert_chunks(client, name, dc, vectors)
                    result["upsert_seconds"] = round(time.perf_counter() - t, 3)
                    result["upserted"] = len(dc)
                except Exception as exc:  # noqa: BLE001 - 문서 하나의 실패가 전체를 멈추지 않게 기록한다
                    result["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                return result

            res = tracer.record(f"index-pdf {doc['file_name']}",
                                {"book_id": doc["book_id"], "chunk_count": len(doc_chunks)},
                                {"book_id": doc["book_id"]}, work)
            per_doc.append(res)
            embed_total += res.get("embed_seconds", 0.0)
            upsert_total += res.get("upsert_seconds", 0.0)
            if "error" in res:
                failed.append(res)
            print(f"  {res['book_id']}: chunks={res['chunks']} upserted={res['upserted']} "
                  f"embed={res.get('embed_seconds')}s {res.get('error', '')}")
    total_seconds = time.perf_counter() - t_all
    if langfuse is not None:
        print(f"Langfuse: 문서 Span {tracer.documents_recorded}개 기록\n  {tracer.trace_url()}")

    # --- 검증 (08 §3.4) ---
    stored = qs.count_points(client, name)
    probe = chunks[len(chunks) // 2]
    hit = qs.search(client, name, embedder.embed_passages([texts[probe["chunk_id"]]])[0], k=1)[0]
    checks = {
        "stored_count": stored, "expected_count": len(chunks), "count_ok": stored == len(chunks),
        "probe_chunk_id": probe["chunk_id"], "probe_top1": hit["chunk_id"],
        "probe_ok": hit["chunk_id"] == probe["chunk_id"] and hit["page"] == probe["page"]
        and hit["book_id"] == probe["book_id"],
        "probe_source": {"file_name": hit["file_name"], "page": hit["page"]},
    }
    print(f"검증: 저장 {stored}/{len(chunks)}, 탐침 {probe['chunk_id']} -> top1 {hit['chunk_id']} "
          f"({hit['file_name']} p.{hit['page']}) ok={checks['probe_ok']}")

    sizes = [len(c["chunk_text"]) for c in chunks]
    manifest = {
        "collection_name": name,
        "qdrant_mode": mode,
        "embedding_model": spec.hf_id,
        "embedding_version": spec.version_tag,
        "embedding_revision": embedder.revision,
        "embedding_license": model_license(spec.hf_id),
        "embedding_dimension": embedder.dimension,
        "max_seq_length": spec.max_seq_length,
        "query_prefix": spec.query_prefix, "passage_prefix": spec.passage_prefix,
        "distance": "cosine", "normalized": True, "device": embedder.device,
        "chunking_version": chunking_version,
        "document_count": len(docs), "chunk_count": len(chunks),
        "failed_chunk_count": sum(r["chunks"] - r["upserted"] for r in per_doc),
        "avg_chunk_size": round(statistics.mean(sizes), 1), "chunk_size_unit": "chars",
        "token_stats": {"unit": "tokens (model tokenizer, prefix included)", "avg": round(statistics.mean(tokens), 1),
                        "max": max(tokens), "over_limit": over},
        "indexing_seconds": {"model_load": round(model_load, 2), "embedding": round(embed_total, 2),
                             "upsert": round(upsert_total, 2), "total_index_loop": round(total_seconds, 2)},
        "per_pdf": [
            {"book_id": d["book_id"], "page_count": d["page_count"], "extraction_status": d["extraction_status"],
             "extraction_seconds": d["extraction_seconds"],
             "chunks": next(r["chunks"] for r in per_doc if r["book_id"] == d["book_id"]),
             "embed_seconds": next(r.get("embed_seconds") for r in per_doc if r["book_id"] == d["book_id"])}
            for d in docs
        ],
        "verification": checks,
        "status": "success" if not failed and checks["count_ok"] and checks["probe_ok"] else "partial",
        "track": "development",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"매니페스트 저장: {manifest_path.name} (status={manifest['status']}, license={manifest['embedding_license']})")
    print(f"시간: 모델 로드 {model_load:.1f}s, 임베딩 {embed_total:.1f}s, 저장 {upsert_total:.1f}s")
    return 0 if manifest["status"] == "success" else 2


if __name__ == "__main__":
    sys.exit(main())
