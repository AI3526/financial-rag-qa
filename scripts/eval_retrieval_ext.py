"""확장 브랜치: 검색만으로 Hit/Recall/MRR/nDCG@5를 계산한다 (모델 호출 없음, Langfuse·Qdrant 미사용, DECISIONS D-047).

사용:
    uv run python scripts/eval_retrieval_ext.py --dataset basic --out ext-basic-retrieval-check-001           # 기존 공식 결과와 대조(검증)
    uv run python scripts/eval_retrieval_ext.py --dataset intermediate --ext --out ext-int-retrieval-base-002   # 기준선
    uv run python scripts/eval_retrieval_ext.py --dataset intermediate --ext --out ext-int-s1 \\
        --grid "20,20,20,20;20,20,40,40;30,30,40,40"                       # 후보 수 설정을 여러 개 한 번에(설정당 폴더 하나)

- 세 방식(Dense, Hybrid, Hybrid+Rerank)을 실행한다. Dense는 numpy 정확 검색이다. 질문 문자열만 검색기에 넣고 평가 파일은 채점에만 쓴다.
- `--grid`의 설정은 `dense후보,BM25후보,fused_top,rerank후보`이다(RRF 상수 `--rrf-k`, 최종 k=5는 고정). Rerank 후보는 fused_top 이하여야 한다.
- 지표: 공식 문서(`book_id`) 단위 Hit/Recall, 쪽 단위 Hit/Recall(참고), 문서·쪽 단위 MRR·nDCG@5. 진단: 재정렬 전 후보 풀의 Recall(상한)과 검색 시간.
- `--ext`: 자체 추가 질문(`evaluation_ext`, 동결됨)을 함께 평가한다. 근거가 없는 문항(주제 부재 abstain)의 지표는 N/A(None)이다.
- 세트: development, additional, ext_dev, ext_hold. `tune`(= development + ext_dev)이 튜닝·선택용, `confirm`(= additional + ext_hold)은 최종 후보 확인용이다.
  기본으로는 tune 결과만 화면에 출력한다(`--show-confirm`을 주면 confirm도 출력). 전체 결과는 파일에 항상 저장된다.
- 결과 폴더가 이미 있으면 덮어쓰지 않고 중단한다.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_env  # noqa: E402

load_env()

from src.evaluation.judge import load_references  # noqa: E402
from src.evaluation.ranking_metrics import ndcg_at_k, reciprocal_rank  # noqa: E402
from src.evaluation.retrieval_metrics import hit_recall, mean_ignoring_none, question_metrics, reference_books, reference_pages  # noqa: E402
from src.index.embedding import Embedder  # noqa: E402
from src.index.numpy_store import NumpyIndex  # noqa: E402
from src.retrieval.dense import RetrievalConfig  # noqa: E402
from src.retrieval.dense_np import NumpyDenseRetriever  # noqa: E402
from src.retrieval.hybrid import HybridConfig, HybridRetriever  # noqa: E402
from src.retrieval.rerank import CrossEncoderReranker, RerankConfig, RerankRetriever  # noqa: E402
from src.retrieval.sparse import SparseRetriever, load_chunks  # noqa: E402

DATASETS = {
    "basic": {"eval": ROOT / "data" / "finance_basic" / "evaluation", "processed": ROOT / "data" / "processed", "index": "basic__chunk-v1__e5large-v1"},
    "intermediate": {"eval": ROOT / "data" / "finance_intermediate" / "evaluation", "processed": ROOT / "data" / "processed_intermediate",
                     "index": "intermediate__chunk-v1__e5large-v1", "ext": ROOT / "data" / "finance_intermediate" / "evaluation_ext"},
}
SETS = (("development", "development.json", "development_questions.json"), ("additional", "additional.json", "additional_questions.json"))
METHODS = ("dense", "hybrid", "hybrid_rerank")
KEYS = ("hit", "recall", "page_hit", "page_recall", "mrr", "page_mrr", "ndcg", "page_ndcg", "cand_hit", "cand_recall", "cand_page_recall")
GROUPS = {"tune": ("development", "ext_dev"), "confirm": ("additional", "ext_hold"), "development": ("development",), "additional": ("additional",),
          "ext_dev": ("ext_dev",), "ext_hold": ("ext_hold",)}


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def aggregate(rows: list[dict]) -> dict:
    """행(질문×방식)들의 평균. 값이 없는(None) 문항은 평균에서 빼고 개수를 적는다."""
    out = {"questions": len(rows)}
    for key in KEYS:
        vals = [r[key] for r in rows if r.get(key) is not None]
        out[key] = {"mean": mean_ignoring_none(vals), "n": len(vals)}
    out["mean_total_ms"] = round(statistics.mean(r["total_ms"] for r in rows), 1) if rows else None
    return out


def parse_grid(text: str) -> list[dict]:
    cfgs = []
    for part in text.split(";"):
        d, s, f, r = (int(x) for x in part.split(","))
        if r > f:
            raise SystemExit(f"Rerank 후보({r})는 fused_top({f}) 이하여야 합니다: {part}")
        cfgs.append({"dense": d, "sparse": s, "fused": f, "rerank": r, "label": f"d{d}s{s}f{f}r{r}"})
    return cfgs


def load_questions(cfg_d: dict, use_ext: bool):
    """(질문 목록, 근거 참고 사전). 질문 목록에는 질문 번호와 질문 문자열, 세트 이름만 담는다."""
    refs = dict(load_references(cfg_d["eval"], tuple(f for _, f, _ in SETS)))
    questions = []
    for set_name, _, q_file in SETS:
        for q in json.loads((cfg_d["eval"] / q_file).read_text(encoding="utf-8")):
            questions.append({"question_id": q["id"], "question": q["question"], "set": set_name})
    if use_ext:
        ext_dir = cfg_d["ext"]
        full = {x["id"]: x for x in json.loads((ext_dir / "ext_full.json").read_text(encoding="utf-8"))}
        for q in json.loads((ext_dir / "ext_questions.json").read_text(encoding="utf-8")):
            refs[q["id"]] = {"evidence_groups": full[q["id"]]["evidence_groups"], "expected_action": full[q["id"]]["expected_action"]}
            questions.append({"question_id": q["id"], "question": q["question"], "set": "ext_" + full[q["id"]]["split"]})
    return questions, refs


def run_config(cfg: dict, args, cfg_d, questions, refs, index, chunks, embedder, sparse, reranker) -> Path:
    out = ROOT / "results" / (args.out if not args.grid else f"{args.out}-{cfg['label']}")
    if out.exists():
        raise FileExistsError(out)
    dense = NumpyDenseRetriever(index, RetrievalConfig(), embedder=embedder)
    hybrid = HybridRetriever(dense, sparse, HybridConfig(dense_candidates=cfg["dense"], sparse_candidates=cfg["sparse"], rrf_k=args.rrf_k, fused_top=cfg["fused"], final_k=5))
    rerank = RerankRetriever(hybrid, reranker, RerankConfig(candidates=cfg["rerank"], final_k=5))
    retrievers = {"dense": dense, "hybrid": hybrid, "hybrid_rerank": rerank}
    rows = []
    for q in questions:
        ev = refs[q["question_id"]]["evidence_groups"]
        rb, rp = reference_books(ev), {(b, p) for b, p in reference_pages(ev)}
        for m in METHODS:
            t = time.perf_counter()
            res = retrievers[m].retrieve(q["question"])
            total_ms = (time.perf_counter() - t) * 1000
            books = [c["book_id"] for c in res.chunks]
            pages = [(c["book_id"], c["page"]) for c in res.chunks]
            row = {"question_id": q["question_id"], "set": q["set"], "expected_action": refs[q["question_id"]]["expected_action"], "rag_method": m,
                   "reference_books": sorted(rb), "reference_pages": sorted([b, p] for b, p in rp),
                   "retrieved": [{"rank": c["rank"], "chunk_id": c["chunk_id"], "book_id": c["book_id"], "page": c["page"]} for c in res.chunks],
                   **question_metrics(res.chunks, ev), "mrr": reciprocal_rank(books, rb, 5), "page_mrr": reciprocal_rank(pages, rp, 5),
                   "ndcg": ndcg_at_k(books, rb, 5), "page_ndcg": ndcg_at_k(pages, rp, 5), "total_ms": round(total_ms, 1)}
            cands = (res.candidates or {}).get("fused") if m != "dense" else None
            if cands:
                row["cand_size"] = len(cands)
                row["cand_hit"], row["cand_recall"] = hit_recall({c["book_id"] for c in cands}, rb)
                _, row["cand_page_recall"] = hit_recall({(c["book_id"], c["page"]) for c in cands}, rp)
            rows.append(row)

    def group(names):
        return {m: aggregate([r for r in rows if r["rag_method"] == m and r["set"] in names]) for m in METHODS}

    present = {r["set"] for r in rows}
    summary = {"dataset": args.dataset, "config": cfg, "rrf_k": args.rrf_k, "questions": len(questions), "k": 5,
               "groups": {g: group(names) for g, names in GROUPS.items() if present & set(names)}}
    manifest = {"dataset": args.dataset, "index": cfg_d["index"], "chunk_count": len(chunks), "config": cfg, "ext_questions": bool(args.ext),
                "retrieval": {"dense": "numpy exact cosine, e5-large, top-5", "hybrid": hybrid.describe(), "hybrid_rerank": rerank.describe()},
                "git_commit": git_commit(), "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out / "retrieval.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    f = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    shown = ["tune"] + (["confirm"] if args.show_confirm else [])
    print(f"[{cfg['label']}] → {out.name}")
    for g in shown:
        if g not in summary["groups"]:
            continue
        for m in METHODS:
            a = summary["groups"][g][m]
            print(f"  {g:7} {m:14} 문서 H{f(a['hit']['mean'])} R{f(a['recall']['mean'])} MRR{f(a['mrr']['mean'])} nDCG{f(a['ndcg']['mean'])} | "
                  f"쪽 H{f(a['page_hit']['mean'])} R{f(a['page_recall']['mean'])} MRR{f(a['page_mrr']['mean'])} nDCG{f(a['page_ndcg']['mean'])} "
                  f"| 후보풀 쪽R{f(a['cand_page_recall']['mean'])} | {a['mean_total_ms']}ms (n={a['hit']['n']}/{a['questions']})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--ext", action="store_true", help="자체 추가 질문(evaluation_ext)을 함께 평가")
    ap.add_argument("--grid", help="후보 수 설정 목록 'dense,bm25,fused,rerank;…' (기본: 20,20,20,20)")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--show-confirm", action="store_true", help="confirm 세트(additional, ext_hold) 결과도 화면에 출력")
    args = ap.parse_args()
    cfg_d = DATASETS[args.dataset]
    if args.ext and "ext" not in cfg_d:
        print("이 데이터셋에는 자체 추가 질문이 없습니다. (중단)")
        return 1
    cfgs = parse_grid(args.grid or "20,20,20,20")
    for c in cfgs:
        target = ROOT / "results" / (args.out if not args.grid else f"{args.out}-{c['label']}")
        if target.exists():
            print(f"이미 결과가 있습니다(덮어쓰지 않음): {target}. (중단)")
            return 1

    questions, refs = load_questions(cfg_d, args.ext)
    chunks = load_chunks(cfg_d["processed"] / "chunks.jsonl")
    index = NumpyIndex.load(cfg_d["index"])
    if [p["chunk_id"] for p in index.payloads] != [c["chunk_id"] for c in chunks]:
        print("색인의 청크 목록이 chunks.jsonl과 다릅니다. (중단)")
        return 1
    embedder = Embedder(RetrievalConfig().spec)
    sparse = SparseRetriever(chunks=chunks)
    reranker = CrossEncoderReranker()
    for c in cfgs:
        run_config(c, args, cfg_d, questions, refs, index, chunks, embedder, sparse, reranker)
    return 0


if __name__ == "__main__":
    sys.exit(main())
