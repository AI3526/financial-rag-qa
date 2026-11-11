"""확장 브랜치: 최종 5개에서 놓친 정답 쪽이 순위 목록의 어디에 있는지 진단한다 (모델 호출 없음).

사용:
    uv run python scripts/diagnose_rank_ext.py --out ext-int-diag-v2b --processed-dir data/processed_intermediate_v2b --index-name intermediate__chunk-v2b-max600__e5large-v1

- 세 방식의 후보 순위 목록(Dense 상위 20, Hybrid 결합 상위 20, Rerank 후보 20을 재정렬한 20)에서, 최종 5개에 못 든 정답 쪽이 몇 번째에 있는지
  (6~10위, 11~20위, 후보 풀 밖)로 센다. 청크가 아니라 정답 쪽이 처음 나오는 순위(같은 쪽의 청크 중 가장 높은 것)를 쓴다.
- 대상은 tune 세트(development + ext_dev)이고 근거가 있는 문항만 센다. 결과는 `results/{out}/diagnosis.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_env  # noqa: E402

load_env()

from src.evaluation.retrieval_metrics import reference_pages  # noqa: E402
from src.index.embedding import Embedder  # noqa: E402
from src.index.numpy_store import NumpyIndex  # noqa: E402
from src.retrieval.dense import RetrievalConfig  # noqa: E402
from src.retrieval.dense_np import NumpyDenseRetriever  # noqa: E402
from src.retrieval.hybrid import HybridConfig, HybridRetriever  # noqa: E402
from src.retrieval.rerank import CrossEncoderReranker, RerankConfig, RerankRetriever  # noqa: E402
from src.retrieval.sparse import SparseRetriever, load_chunks  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from eval_retrieval_ext import DATASETS, GROUPS, load_questions  # noqa: E402


def bucket(rank: int | None) -> str:
    if rank is None:
        return "후보 풀 밖"
    if rank <= 5:
        return "1~5위(최종에 포함)"
    return "6~10위" if rank <= 10 else "11~20위"


def first_rank_of_pages(chunks: list[dict], pages: set) -> dict:
    """정답 쪽마다 목록에서 처음 나오는 순위(없으면 None)."""
    out = {p: None for p in pages}
    for i, c in enumerate(chunks, start=1):
        key = (c["book_id"], c["page"])
        if key in out and out[key] is None:
            out[key] = i
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--processed-dir")
    ap.add_argument("--index-name")
    args = ap.parse_args()
    out = ROOT / "results" / args.out
    if out.exists():
        print(f"이미 결과가 있습니다(덮어쓰지 않음): {out}. (중단)")
        return 1
    cfg_d = dict(DATASETS["intermediate"])
    if args.processed_dir:
        cfg_d["processed"], cfg_d["index"] = ROOT / args.processed_dir, args.index_name
    questions, refs = load_questions(cfg_d, True)
    questions = [q for q in questions if q["set"] in GROUPS["tune"]]
    chunks = load_chunks(cfg_d["processed"] / "chunks.jsonl")
    index = NumpyIndex.load(cfg_d["index"])
    embedder = Embedder(RetrievalConfig().spec)
    dense = NumpyDenseRetriever(index, RetrievalConfig(), embedder=embedder)
    hybrid = HybridRetriever(dense, SparseRetriever(chunks=chunks), HybridConfig(dense_candidates=20, sparse_candidates=20, fused_top=20, final_k=5))
    rerank = RerankRetriever(hybrid, CrossEncoderReranker(), RerankConfig(candidates=20, final_k=5))
    retrievers = {"dense": dense, "hybrid": hybrid, "hybrid_rerank": rerank}
    counts = {m: Counter() for m in retrievers}
    per_question = []
    for q in questions:
        pages = {tuple(p) for p in reference_pages(refs[q["question_id"]]["evidence_groups"])}
        if not pages:
            continue
        for m, r in retrievers.items():
            ranks = first_rank_of_pages(r.retrieve(q["question"], k=20).chunks, pages)
            for page, rank in ranks.items():
                counts[m][bucket(rank)] += 1
                if rank is None or rank > 5:
                    per_question.append({"question_id": q["question_id"], "method": m, "page": list(page), "rank": rank})
    out.mkdir(parents=True)
    total = {m: sum(c.values()) for m, c in counts.items()}
    summary = {"tune_questions_with_evidence": len({x["question_id"] for x in per_question} | set()), "reference_pages_total": total,
               "buckets": {m: dict(c) for m, c in counts.items()}, "missed_pages": per_question}
    (out / "diagnosis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    order = ["1~5위(최종에 포함)", "6~10위", "11~20위", "후보 풀 밖"]
    for m in retrievers:
        print(f"{m:14} 정답 쪽 {total[m]}개: " + ", ".join(f"{b} {counts[m][b]}({counts[m][b] / total[m]:.0%})" for b in order))
    return 0


if __name__ == "__main__":
    sys.exit(main())
