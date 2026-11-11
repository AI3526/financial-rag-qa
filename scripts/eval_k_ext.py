"""확장 브랜치: 후보 풀 크기 × 최종 청크 수(k) 조합 평가 (Hybrid+Rerank, chunk-v1, 새 프롬프트·LLM 호출 없음).

배경(EXTENSION_INTERMEDIATE_LOG 진단): 답변 실패 문항의 정답 청크가 후보 20 안에 있어도 재정렬 7~14위라 최종 5개에서 잘리는 경우가 있었다.
후보 풀마다 재정렬 결과 전체(순위·점수·정답 쪽 여부)를 저장하고, k를 바꿔 같은 순위에서 지표를 계산한다(재정렬은 풀당 1회).
- 지표: 문서 Hit/Recall, 쪽 Hit/Recall, 근거 문장 포함(quote), 사실 숫자 포함률, 최종 문맥 글자 수.
- 판단은 tune 세트(development + ext_dev)에서만 한다. confirm 세트(additional, ext_hold)도 계산해 저장하지만 화면에는 내지 않는다.
- 정답 파일은 채점에만 쓴다.

사용: uv run python scripts/eval_k_ext.py --out ext-int-k-base-001
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.config import load_env  # noqa: E402

load_env()

from decompose_queries_ext import load_questions  # noqa: E402
from fact_coverage_ext import TUNE, coverage, load_refs  # noqa: E402
from src.evaluation.compare import paired_stats  # noqa: E402
from src.evaluation.ranking_metrics import quote_coverage  # noqa: E402
from src.evaluation.retrieval_metrics import question_metrics, reference_pages  # noqa: E402
from src.index.embedding import Embedder  # noqa: E402
from src.index.numpy_store import NumpyIndex  # noqa: E402
from src.retrieval.dense import RetrievalConfig  # noqa: E402
from src.retrieval.dense_np import NumpyDenseRetriever  # noqa: E402
from src.retrieval.hybrid import HybridConfig, HybridRetriever  # noqa: E402
from src.retrieval.rerank import CrossEncoderReranker, RerankConfig, RerankRetriever  # noqa: E402
from src.retrieval.sparse import SparseRetriever, load_chunks  # noqa: E402

POOLS = {"p20": (20, 20, 20, 20), "p40": (20, 20, 40, 40), "p60": (40, 40, 60, 60)}  # 이전 실험과 같은 후보 수 설정(dense, bm25, fused, rerank)
KS = (5, 8, 10, 12, 15)
METRICS = ("hit", "recall", "page_hit", "page_recall", "quote_recall", "fact_cov", "chars")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / "results" / args.out
    if out.exists():
        print(f"이미 결과가 있습니다(덮어쓰지 않음): {out}. (중단)")
        return 1
    refs = load_refs()
    chunks = load_chunks(ROOT / "data" / "processed_intermediate" / "chunks.jsonl")
    index = NumpyIndex.load("intermediate__chunk-v1__e5large-v1")
    dense = NumpyDenseRetriever(index, RetrievalConfig(), embedder=Embedder(RetrievalConfig().spec))
    sparse = SparseRetriever(chunks=chunks)
    reranker = CrossEncoderReranker()
    questions = [q for q in load_questions() if refs[q["question_id"]]["evidence_groups"]]  # 근거가 있는 질문만(abstain·clarify 제외)
    rows = []
    ranked_dump = []
    for pname, (d, s, f, r) in POOLS.items():
        hybrid = HybridRetriever(dense, sparse, HybridConfig(dense_candidates=d, sparse_candidates=s, fused_top=f, final_k=5))
        rr = RerankRetriever(hybrid, reranker, RerankConfig(candidates=r, final_k=5))
        for q in questions:
            ref = refs[q["question_id"]]
            ev, facts = ref["evidence_groups"], ref["required_facts"]
            ref_pages = {(b, p) for b, p in reference_pages(ev)}
            t = time.perf_counter()
            res = rr.retrieve(q["question"], k=r)
            ms = (time.perf_counter() - t) * 1000
            ranked_dump.append({"pool": pname, "question_id": q["question_id"], "set": q["set"],
                                "ranked": [{"rank": c["rank"], "chunk_id": c["chunk_id"], "book_id": c["book_id"], "page": c["page"], "score": round(c["score"], 4),
                                            "gold_page": (c["book_id"], c["page"]) in ref_pages, "chars": len(c["text"])} for c in res.chunks]})
            for k in KS:
                top = res.chunks[:k]
                m = question_metrics(top, ev)
                qh, qr = quote_coverage(top, ev)
                rows.append({"pool": pname, "k": k, "question_id": q["question_id"], "set": q["set"], "hit": m["hit"], "recall": m["recall"], "page_hit": m["page_hit"],
                             "page_recall": m["page_recall"], "quote_recall": qr, "fact_cov": coverage("".join(c["text"] for c in top), facts),
                             "chars": sum(len(c["text"]) for c in top), "ms": round(ms, 1)})
    out.mkdir(parents=True)
    (out / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out / "ranked.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ranked_dump), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"pools": POOLS, "ks": KS, "index": "intermediate__chunk-v1__e5large-v1", "method": "hybrid_rerank",
                                                  "questions": len(questions)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # 화면에는 tune만 출력한다.
    tune = [r for r in rows if r["set"] in TUNE]
    base = {r["question_id"]: r for r in tune if r["pool"] == "p20" and r["k"] == 5}

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    print(f"tune {len(base)}문항, Hybrid+Rerank. 기준선 = 풀 20, k=5")
    print(f"{'풀':4}{'k':>3} | " + " ".join(f"{m:>11}" for m in METRICS) + " | 쪽Recall 차이(기준선 대비) 좋/나쁨/같음")
    for pname in POOLS:
        for k in KS:
            sel = {r["question_id"]: r for r in tune if r["pool"] == pname and r["k"] == k}
            cells = [mean([r[m] for r in sel.values()]) for m in METRICS]
            st = paired_stats({q: base[q]["page_recall"] for q in base}, {q: sel[q]["page_recall"] for q in base})
            print(f"{pname:4}{k:>3} | " + " ".join(f"{c:>11.3f}" if m != "chars" else f"{c:>11.0f}" for c, m in zip(cells, METRICS))
                  + f" | {st['mean_diff']:+.3f} {st['better']}/{st['worse']}/{st['same']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
