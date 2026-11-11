"""확장 브랜치: numpy Dense 검색이 Basic의 Qdrant Dense 검색과 같은 결과를 내는지 확인한다 (DECISIONS D-047).

사용: uv run python scripts/verify_numpy_dense.py --out ext-verify-numpy-001

- Basic 60문항(development 30 + additional 30)을 질문 벡터 하나로 두 검색기(Qdrant Cloud 읽기 전용, numpy)에 넣어 상위 5개를 비교한다.
- 질문 벡터를 공유하므로 차이는 저장된 문서 벡터와 검색 알고리즘에서만 생긴다. Qdrant에 쓰지 않고 Langfuse도 쓰지 않는다.
- 결과 폴더가 이미 있으면 중단한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_env  # noqa: E402

load_env()

from src.evaluation.official import load_question_sets  # noqa: E402
from src.index.embedding import Embedder  # noqa: E402
from src.index.numpy_store import NumpyIndex  # noqa: E402
from src.retrieval.dense import DenseRetriever, RetrievalConfig  # noqa: E402
from src.retrieval.dense_np import NumpyDenseRetriever  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--index", default="basic__chunk-v1__e5large-v1")
    args = ap.parse_args()
    out = ROOT / "results" / args.out
    if out.exists():
        print(f"이미 결과가 있습니다(덮어쓰지 않음): {out}. (중단)")
        return 1
    cfg = RetrievalConfig()
    embedder = Embedder(cfg.spec)
    qdrant = DenseRetriever(cfg, embedder=embedder)
    numpy_r = NumpyDenseRetriever(NumpyIndex.load(args.index), cfg, embedder=embedder)
    rows, same_ids, same_order, max_diff = [], 0, 0, 0.0
    questions = load_question_sets()
    for q in questions:
        a, b = qdrant.retrieve(q["question"]), numpy_r.retrieve(q["question"])
        ia, ib = [c["chunk_id"] for c in a.chunks], [c["chunk_id"] for c in b.chunks]
        diff = max(abs(x["score"] - y["score"]) for x, y in zip(a.chunks, b.chunks))
        same_ids += set(ia) == set(ib)
        same_order += ia == ib
        max_diff = max(max_diff, diff)
        rows.append({"question_id": q["question_id"], "same_set": set(ia) == set(ib), "same_order": ia == ib, "max_score_diff": round(diff, 6),
                     "qdrant": ia, "numpy": ib})
    summary = {"questions": len(questions), "same_top5_set": same_ids, "same_top5_order": same_order, "max_score_diff": round(max_diff, 6),
               "index": args.index, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out / "comparison.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
    for r in rows:
        if not r["same_order"]:
            print("  다름:", r["question_id"], "집합 같음" if r["same_set"] else "집합 다름", r["max_score_diff"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
