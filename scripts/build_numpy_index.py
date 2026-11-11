"""확장 브랜치: chunks.jsonl을 임베딩해 numpy 정확 검색 색인을 만든다 (DECISIONS D-047).

사용:
    uv run python scripts/build_numpy_index.py --processed data/processed --model e5
    uv run python scripts/build_numpy_index.py --processed data/processed_intermediate --model e5

- 입력: `{processed}/chunks.jsonl`과 `chunking_manifest.json`(STEP 1·2 결과). 평가 파일은 읽지 않는다.
- 출력: `data/numpy_index/{dataset}__{chunking_version}__{임베딩버전}/` (vectors.npy, payloads.jsonl, meta.json). 이미 있으면 중단한다.
- Qdrant·Langfuse를 쓰지 않는다. 임베딩 모델·접두어·정규화는 STEP 3과 같다(`src.index.embedding`).
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

load_env()

from src.index.embedding import CANDIDATES, Embedder, collection_name  # noqa: E402
from src.index.numpy_store import NumpyIndex  # noqa: E402
from src.ingest import chunk as ck  # noqa: E402
from src.ingest.extract import read_jsonl  # noqa: E402

BATCH = 64


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed", help="chunks.jsonl이 있는 폴더(저장소 루트 기준)")
    ap.add_argument("--model", default="e5", choices=sorted(CANDIDATES))
    args = ap.parse_args()
    spec = CANDIDATES[args.model]
    processed = (ROOT / args.processed).resolve()
    chunks = read_jsonl(processed / "chunks.jsonl")
    chunking = json.loads((processed / "chunking_manifest.json").read_text(encoding="utf-8"))
    name = collection_name(chunking["dataset_id"], chunking["chunking_version"], spec)

    t0 = time.perf_counter()
    embedder = Embedder(spec)
    load_s = time.perf_counter() - t0
    print(f"{name}: 청크 {len(chunks)}개, 모델 로드 {load_s:.1f}s device={embedder.device} revision={embedder.revision}")
    texts = [ck.retrieval_text(c) for c in chunks]
    t1 = time.perf_counter()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        vectors += embedder.embed_passages(texts[i:i + BATCH])
        if (i // BATCH) % 20 == 0:
            print(f"  {min(i + BATCH, len(texts))}/{len(texts)}", flush=True)
    embed_s = time.perf_counter() - t1
    sizes = [len(c["chunk_text"]) for c in chunks]
    meta = {"name": name, "embedding_model": spec.hf_id, "embedding_version": spec.version_tag, "embedding_revision": embedder.revision,
            "query_prefix": spec.query_prefix, "passage_prefix": spec.passage_prefix, "distance": "cosine", "normalized": True,
            "search": "numpy exact (float32 matrix product, stable argsort)", "chunking_version": chunking["chunking_version"],
            "dataset_id": chunking["dataset_id"], "processed_dir": str(processed.relative_to(ROOT)), "chunk_count": len(chunks),
            "avg_chunk_size_chars": round(statistics.mean(sizes), 1), "embed_seconds": round(embed_s, 1), "model_load_seconds": round(load_s, 1),
            "device": embedder.device, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    index = NumpyIndex.build(chunks, vectors, meta)
    try:
        out = index.save(name)
    except FileExistsError as e:
        print(e, "(중단)")
        return 1
    print(f"저장: {out} (행렬 {index.vectors.shape}, {index.vectors.nbytes / 1e6:.1f}MB, 임베딩 {embed_s:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
