"""확장 브랜치: 검색만으로 Hit@k·Recall@k를 계산한다 (모델 호출 없음, Langfuse·Qdrant 미사용, DECISIONS D-047).

사용:
    uv run python scripts/eval_retrieval_ext.py --dataset basic --out ext-basic-retrieval-check-001       # 기존 공식 결과와 대조(검증)
    uv run python scripts/eval_retrieval_ext.py --dataset intermediate --out ext-int-retrieval-base-001   # 기준선

- 세 방식(Dense, Hybrid = Dense20 + BM25 20 → RRF → 상위 20 → 최종 5, Hybrid+Rerank)을 Basic과 같은 설정으로 실행한다. Dense는 numpy 정확 검색이다.
- 질문 문자열만 검색기에 넣는다(`*_questions.json`). 평가 파일(`evidence_groups`)은 결과를 채점하는 데만 쓴다.
- 공식 지표는 `book_id`(문서) 단위이고 `(book_id, page)` 단위는 참고용이다(`src.evaluation.retrieval_metrics`와 같은 계산).
- 진단용으로 Hybrid·Rerank의 후보 20개(재정렬 전)에 근거가 들어 있는 비율도 센다(Rerank가 개선할 수 있는 상한).
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
from src.evaluation.retrieval_metrics import hit_recall, mean_ignoring_none, question_metrics, reference_books, reference_pages  # noqa: E402
from src.index.embedding import CANDIDATES, Embedder  # noqa: E402
from src.index.numpy_store import NumpyIndex  # noqa: E402
from src.retrieval.dense import RetrievalConfig  # noqa: E402
from src.retrieval.dense_np import NumpyDenseRetriever  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402
from src.retrieval.rerank import CrossEncoderReranker, RerankRetriever  # noqa: E402
from src.retrieval.sparse import SparseRetriever, load_chunks  # noqa: E402

DATASETS = {
    "basic": {"eval": ROOT / "data" / "finance_basic" / "evaluation", "processed": ROOT / "data" / "processed", "index": "basic__chunk-v1__e5large-v1"},
    "intermediate": {"eval": ROOT / "data" / "finance_intermediate" / "evaluation", "processed": ROOT / "data" / "processed_intermediate",
                     "index": "intermediate__chunk-v1__e5large-v1"},
}
SETS = (("development", "development.json", "development_questions.json"), ("additional", "additional.json", "additional_questions.json"))
METHODS = ("dense", "hybrid", "hybrid_rerank")


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def aggregate(rows: list[dict]) -> dict:
    """행(질문×방식)들의 평균. 값이 없는(None) 문항은 평균에서 빼고 개수를 적는다."""
    out = {"questions": len(rows)}
    for key in ("hit", "recall", "page_hit", "page_recall", "cand_hit", "cand_recall", "cand_page_recall"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        out[key] = {"mean": mean_ignoring_none(vals), "n": len(vals)}
    out["mean_total_ms"] = round(statistics.mean(r["total_ms"] for r in rows), 1) if rows else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / "results" / args.out
    if out.exists():
        print(f"이미 결과가 있습니다(덮어쓰지 않음): {out}. (중단)")
        return 1
    cfg_d = DATASETS[args.dataset]

    refs = load_references(cfg_d["eval"], tuple(f for _, f, _ in SETS))
    set_of, questions = {}, []
    for set_name, ref_file, q_file in SETS:
        for q in json.loads((cfg_d["eval"] / q_file).read_text(encoding="utf-8")):
            set_of[q["id"]] = set_name
            questions.append({"question_id": q["id"], "question": q["question"], "set": set_name})  # 질문 번호와 질문만 읽는다
    chunks = load_chunks(cfg_d["processed"] / "chunks.jsonl")
    index = NumpyIndex.load(cfg_d["index"])
    if [p["chunk_id"] for p in index.payloads] != [c["chunk_id"] for c in chunks]:
        print("색인의 청크 목록이 chunks.jsonl과 다릅니다. (중단)")
        return 1

    cfg = RetrievalConfig()
    embedder = Embedder(cfg.spec)
    dense = NumpyDenseRetriever(index, cfg, embedder=embedder)
    sparse = SparseRetriever(chunks=chunks)
    hybrid = HybridRetriever(dense, sparse)
    reranker = CrossEncoderReranker()
    rerank = RerankRetriever(hybrid, reranker)
    retrievers = {"dense": dense, "hybrid": hybrid, "hybrid_rerank": rerank}

    rows = []
    for q in questions:
        ev = refs[q["question_id"]]["evidence_groups"]
        for m in METHODS:
            t = time.perf_counter()
            res = retrievers[m].retrieve(q["question"])
            total_ms = (time.perf_counter() - t) * 1000
            row = {"question_id": q["question_id"], "set": q["set"], "expected_action": refs[q["question_id"]]["expected_action"], "rag_method": m,
                   "reference_books": sorted(reference_books(ev)), "reference_pages": sorted([b, p] for b, p in reference_pages(ev)),
                   "retrieved": [{"rank": c["rank"], "chunk_id": c["chunk_id"], "book_id": c["book_id"], "page": c["page"]} for c in res.chunks],
                   **question_metrics(res.chunks, ev), "total_ms": round(total_ms, 1)}
            cands = (res.candidates or {}).get("fused") if m != "dense" else None
            if cands:  # 재정렬·결합 전 후보 20개에 근거가 있는 비율(상한 진단)
                row["cand_hit"], row["cand_recall"] = hit_recall({c["book_id"] for c in cands}, reference_books(ev))
                _, row["cand_page_recall"] = hit_recall({(c["book_id"], c["page"]) for c in cands}, reference_pages(ev))
            rows.append(row)

    def group(pred):
        return {m: aggregate([r for r in rows if r["rag_method"] == m and pred(r)]) for m in METHODS}

    summary = {"dataset": args.dataset, "questions": len(questions), "k": 5, "overall": group(lambda r: True),
               "by_set": {s: group(lambda r, s=s: r["set"] == s) for s, _, _ in SETS},
               "by_expected_action": {a: group(lambda r, a=a: r["expected_action"] == a) for a in sorted({r["expected_action"] for r in rows})}}
    manifest = {"dataset": args.dataset, "index": cfg_d["index"], "index_meta": index.meta, "chunk_count": len(chunks),
                "retrieval": {"dense": "numpy exact cosine, e5-large, top-5", "hybrid": hybrid.describe(), "hybrid_rerank": rerank.describe()},
                "git_commit": git_commit(), "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out / "retrieval.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    f = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    print(f"{args.dataset}: 질문 {len(questions)}개, k=5 → {out}")
    for m in METHODS:
        a = summary["overall"][m]
        print(f"  {m:14} 문서 Hit {f(a['hit']['mean'])} Recall {f(a['recall']['mean'])} (n={a['hit']['n']}) | 쪽 Hit {f(a['page_hit']['mean'])} Recall {f(a['page_recall']['mean'])}"
              f" | 후보20 문서 Recall {f(a['cand_recall']['mean'])} 쪽 Recall {f(a['cand_page_recall']['mean'])} | {a['mean_total_ms']}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
