"""확장 브랜치: 후보 풀 밖의 정답 쪽이 '왜' 못 찾히는지 진단한다 (모델 호출 없음).

사용:
    uv run python scripts/diagnose_missed_ext.py --out ext-int-diag-missed-v2b --processed-dir data/processed_intermediate_v2b --index-name intermediate__chunk-v2b-max600__e5large-v1

각 정답 쪽(tune, 근거 있는 문항)에 대해:
- dense_rank / bm25_rank: **전체 청크**에서 그 쪽의 가장 좋은 청크가 몇 번째인가(후보 20개 안에 못 든 이유가 '조금 모자람'인지 '아주 멀리 있음'인지).
- same_quote_pages: 같은 근거 문장이 들어 있는 쪽이 코퍼스에 몇 개인가(같은 내용이 여러 쪽에 반복되면 특정 쪽을 못 찾아도 근거는 찾은 것일 수 있다).
- quote_in_pool / quote_in_final5: 근거 문장이 (다른 쪽이라도) Hybrid 후보 풀 20개 또는 최종 5개 안에 있는가.
- overlap: 질문의 어절 중 근거 문장에 그대로 나오는 비율(질문과 근거 문장이 말을 얼마나 공유하는가).
정답 쪽을 '찾은 쪽'(최종 5개 안)과 '놓친 쪽'(풀 밖)으로 나눠 비교한다. 결과는 `results/{out}/diagnosis.json`.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_env  # noqa: E402

load_env()

from src.index.embedding import Embedder  # noqa: E402
from src.index.numpy_store import NumpyIndex  # noqa: E402
from src.retrieval.dense import RetrievalConfig  # noqa: E402
from src.retrieval.dense_np import NumpyDenseRetriever  # noqa: E402
from src.retrieval.hybrid import HybridConfig, HybridRetriever  # noqa: E402
from src.retrieval.rerank import CrossEncoderReranker, RerankConfig, RerankRetriever  # noqa: E402
from src.retrieval.sparse import SparseRetriever, load_chunks  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from eval_retrieval_ext import DATASETS, GROUPS, load_questions  # noqa: E402


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def best_rank(scores: np.ndarray, indices: list[int]) -> int:
    """indices 중 점수가 가장 높은 청크가 전체에서 몇 번째인가(1부터)."""
    top = max(scores[i] for i in indices)
    return int((scores > top).sum()) + 1


def overlap(question: str, quote: str) -> float:
    toks = [t for t in question.split() if len(t) >= 2]
    return sum(1 for t in toks if t in quote) / len(toks) if toks else 0.0


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
    by_page: dict[tuple, list[int]] = {}
    for i, c in enumerate(chunks):
        by_page.setdefault((c["book_id"], c["page"]), []).append(i)
    page_text = {k: norm(" ".join(chunks[i]["chunk_text"] for i in v)) for k, v in by_page.items()}
    embedder = Embedder(RetrievalConfig().spec)
    dense = NumpyDenseRetriever(index, RetrievalConfig(), embedder=embedder)
    sparse = SparseRetriever(chunks=chunks)
    hybrid = HybridRetriever(dense, sparse, HybridConfig(dense_candidates=20, sparse_candidates=20, fused_top=20, final_k=5))
    rerank = RerankRetriever(hybrid, CrossEncoderReranker(), RerankConfig(candidates=20, final_k=5))
    rows = []
    for q in questions:
        groups = refs[q["question_id"]]["evidence_groups"]
        if not any(groups):
            continue
        qvec = np.asarray(embedder.embed_queries([q["question"]])[0], dtype=np.float32)
        dscores = index.vectors @ qvec
        bscores = np.asarray(sparse.bm25.get_scores(sparse.tokenize(q["question"])))
        pool = hybrid.retrieve(q["question"], k=20).chunks
        final = rerank.retrieve(q["question"], k=5).chunks
        pool_pages, final_pages = {(c["book_id"], c["page"]) for c in pool}, {(c["book_id"], c["page"]) for c in final}
        pool_text, final_text = norm(" ".join(c["text"] for c in pool)), norm(" ".join(c["text"] for c in final))
        seen = set()
        for g in groups:
            for e in g:
                key = (e["book_id"], e["page"])
                if key in seen:
                    continue
                seen.add(key)
                quote = norm(e["quote"])
                rows.append({"question_id": q["question_id"], "page": list(key), "status": "found" if key in final_pages else ("pool_only" if key in pool_pages else "missed"),
                             "dense_rank": best_rank(dscores, by_page[key]), "bm25_rank": best_rank(bscores, by_page[key]),
                             "same_quote_pages": sum(1 for t in page_text.values() if quote in t), "quote_in_pool": quote in pool_text, "quote_in_final5": quote in final_text,
                             "overlap": round(overlap(q["question"], e["quote"]), 3)})
    out.mkdir(parents=True)
    (out / "diagnosis.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for status in ("found", "pool_only", "missed"):
        r = [x for x in rows if x["status"] == status]
        if not r:
            continue
        d, b = [x["dense_rank"] for x in r], [x["bm25_rank"] for x in r]
        print(f"[{status}] 정답 쪽 {len(r)}개 | dense 순위 중앙 {statistics.median(d):.0f} (≤20 {sum(v <= 20 for v in d)}, ≤100 {sum(v <= 100 for v in d)}, ≤500 {sum(v <= 500 for v in d)}) | "
              f"bm25 순위 중앙 {statistics.median(b):.0f} (≤20 {sum(v <= 20 for v in b)}, ≤100 {sum(v <= 100 for v in b)}, ≤500 {sum(v <= 500 for v in b)}) | "
              f"질문-근거 어절 겹침 평균 {statistics.mean(x['overlap'] for x in r):.2f} | 같은 근거 문장이 든 쪽 수 중앙 {statistics.median(x['same_quote_pages'] for x in r):.0f} | "
              f"근거 문장이 풀에 있음 {sum(x['quote_in_pool'] for x in r)} / 최종 5개에 있음 {sum(x['quote_in_final5'] for x in r)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
