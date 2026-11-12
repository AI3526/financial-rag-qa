"""확장 브랜치: 질의별 자리 배정 검색(SlotQueryRerank)의 검색 단계 평가 — LLM 호출 없음(분해 질의는 저장된 결과 사용).

기준선은 `ext-int-k-base-001`의 후보 40(`p40`)·최종 12개(k=12) 행이다(같은 후보 수·같은 최종 청크 수). 판단은 tune 세트(development + ext_dev)에서만 하고
confirm 세트는 계산해 저장하되 화면에 내지 않는다. 지표는 eval_k_ext.py와 같다.

사용: uv run python scripts/eval_slots_ext.py --out ext-int-slot-001 --queries-file ext-int-s5-queries-v1
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
from src.evaluation.retrieval_metrics import question_metrics  # noqa: E402
from src.index.embedding import Embedder  # noqa: E402
from src.index.numpy_store import NumpyIndex  # noqa: E402
from src.retrieval.dense import RetrievalConfig  # noqa: E402
from src.retrieval.dense_np import NumpyDenseRetriever  # noqa: E402
from src.retrieval.hybrid import HybridConfig, HybridRetriever  # noqa: E402
from src.retrieval.rerank import CrossEncoderReranker, RerankConfig, RerankRetriever  # noqa: E402
from src.retrieval.slot_query import SlotQueryRerank  # noqa: E402
from src.retrieval.sparse import SparseRetriever, load_chunks  # noqa: E402

K = 12
METRICS = ("hit", "recall", "page_hit", "page_recall", "quote_recall", "fact_cov", "chars")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--queries-file", required=True, help="분해 질의가 저장된 결과 폴더(decompose_queries_ext.py)")
    ap.add_argument("--sub-slots", type=int, default=3)
    ap.add_argument("--sub-cap", type=int, default=6)
    args = ap.parse_args()
    out = ROOT / "results" / args.out
    if out.exists():
        print(f"이미 결과가 있습니다(덮어쓰지 않음): {out}. (중단)")
        return 1
    refs = load_refs()
    lookup = {}
    for line in (ROOT / "results" / args.queries_file / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            lookup[r["question"]] = r["queries"]
    chunks = load_chunks(ROOT / "data" / "processed_intermediate" / "chunks.jsonl")
    index = NumpyIndex.load("intermediate__chunk-v1__e5large-v1")
    dense = NumpyDenseRetriever(index, RetrievalConfig(), embedder=Embedder(RetrievalConfig().spec))
    hybrid = HybridRetriever(dense, SparseRetriever(chunks=chunks), HybridConfig(dense_candidates=20, sparse_candidates=20, fused_top=40, final_k=K))
    rr = RerankRetriever(hybrid, CrossEncoderReranker(), RerankConfig(candidates=40, final_k=K))
    slot = SlotQueryRerank(rr, lookup, pool=40, sub_slots=args.sub_slots, sub_cap=args.sub_cap)
    rows = []
    for q in (x for x in load_questions() if refs[x["question_id"]]["evidence_groups"]):
        ref = refs[q["question_id"]]
        t = time.perf_counter()
        res = slot.retrieve(q["question"], k=K)
        ms = (time.perf_counter() - t) * 1000
        m = question_metrics(res.chunks, ref["evidence_groups"])
        qh, qr = quote_coverage(res.chunks, ref["evidence_groups"])
        rows.append({"question_id": q["question_id"], "set": q["set"], "extra_queries": len(lookup.get(q["question"], [])), "hit": m["hit"], "recall": m["recall"],
                     "page_hit": m["page_hit"], "page_recall": m["page_recall"], "quote_recall": qr, "fact_cov": coverage("".join(c["text"] for c in res.chunks), ref["required_facts"]),
                     "chars": sum(len(c["text"]) for c in res.chunks), "ms": round(ms, 1),
                     "retrieved": [{"rank": c["rank"], "chunk_id": c["chunk_id"], "book_id": c["book_id"], "page": c["page"], "slot": c["slot_source"]} for c in res.chunks]})
    out.mkdir(parents=True)
    (out / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"method": "slot_query_rerank", "k": K, "pool": 40, "sub_slots": args.sub_slots, "sub_cap": args.sub_cap, "queries_file": args.queries_file,
                                                  "baseline": "ext-int-k-base-001 p40 k=12", "describe": slot.describe()}, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    base = {json.loads(l)["question_id"]: json.loads(l) for l in (ROOT / "results" / "ext-int-k-base-001" / "rows.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip() and json.loads(l)["pool"] == "p40" and json.loads(l)["k"] == K}
    tune = [r for r in rows if r["set"] in TUNE]
    mean = lambda vals: (sum(v for v in vals if v is not None) / len([v for v in vals if v is not None])) if any(v is not None for v in vals) else float("nan")  # noqa: E731
    print(f"tune {len(tune)}문항, 후보 40 · 최종 {K}개. 기준선 = 분해 없음(후보 40, k=12)")
    print(f"{'':10} " + " ".join(f"{m:>11}" for m in METRICS) + " | 평균 검색 ms")
    print(f"{'기준선':10} " + " ".join(f"{mean([base[r['question_id']][m] for r in tune]):>11.3f}" if m != "chars" else f"{mean([base[r['question_id']][m] for r in tune]):>11.0f}" for m in METRICS) + f" | {mean([base[r['question_id']]['ms'] for r in tune]):.0f}")
    print(f"{'자리 배정':10} " + " ".join(f"{mean([r[m] for r in tune]):>11.3f}" if m != "chars" else f"{mean([r[m] for r in tune]):>11.0f}" for m in METRICS) + f" | {mean([r['ms'] for r in tune]):.0f}")
    for m in ("page_recall", "quote_recall", "fact_cov", "hit", "recall"):
        st = paired_stats({r["question_id"]: base[r["question_id"]][m] for r in tune}, {r["question_id"]: r[m] for r in tune})
        print(f"  {m:13} 차이 {st['mean_diff']:+.3f} 좋아짐/나빠짐/같음 {st['better']}/{st['worse']}/{st['same']}")
    print("추가 질의가 있는 질문:", sum(1 for r in tune if r["extra_queries"]), "/", len(tune))
    return 0


if __name__ == "__main__":
    sys.exit(main())
