"""STEP 5: Dense, BM25, Hybrid 검색 결과를 같은 development 질문으로 비교한다 (08 §9.3).

사용: uv run python scripts/compare_retrieval.py

- 검색만 비교한다(생성 호출 없음). 최종 k=5, 지표는 `book_id` 단위 Hit@5·Recall@5(공식)와 `(book_id, page)` 단위(참고).
- Hybrid는 동결된 초기값(Dense 20 + BM25 20 → RRF k=60 → 상위 20 → 최종 5)을 그대로 쓴다. 설정 조정은 [확장]이다.
- 질문 문자열만 검색에 쓰고 정답은 검색 뒤 지표 계산에만 쓴다. 로컬에만 저장한다(Langfuse 미사용, D-029).
- 결과 폴더가 있으면 덮어쓰지 않고 중단한다. 승자 선정이 아니라 비교 기록이다(최종 선택은 STEP 12 공식 비교).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_env  # noqa: E402

load_env()

from src.evaluation.retrieval_compare import evaluate_retriever  # noqa: E402
from src.retrieval.dense import DenseRetriever  # noqa: E402
from src.retrieval.hybrid import HybridConfig, HybridRetriever  # noqa: E402
from src.retrieval.sparse import SparseRetriever  # noqa: E402

EXPERIMENT_ID = "dev-retrieval-compare-001"
EVAL_DIR = ROOT / "data" / "finance_basic" / "evaluation"


def main() -> int:
    out = ROOT / "results" / EXPERIMENT_ID
    if out.exists():
        print(f"이미 결과가 있습니다(덮어쓰지 않음): {out}. (중단)")
        return 1
    questions = json.loads((EVAL_DIR / "development_questions.json").read_text(encoding="utf-8"))
    answers = {a["id"]: a for a in json.loads((EVAL_DIR / "development.json").read_text(encoding="utf-8"))}
    dense, sparse = DenseRetriever(), SparseRetriever()
    hybrid = HybridRetriever(dense, sparse)
    methods = {"dense": lambda q: dense.retrieve(q, k=5).chunks, "bm25": lambda q: sparse.retrieve(q, k=5).chunks,
               "hybrid": lambda q: hybrid.retrieve(q).chunks}
    results, seconds = {}, {}
    for name, fn in methods.items():
        t = time.perf_counter()
        results[name] = evaluate_retriever(fn, questions, answers)
        seconds[name] = round(time.perf_counter() - t, 2)
        s = results[name]["summary"]
        print(f"{name:7} Hit@5={s['hit_at_5']:.4f} Recall@5={s['recall_at_5']:.4f} | 페이지 단위 Hit={s['page_hit_at_5']:.4f} "
              f"Recall={s['page_recall_at_5']:.4f} | 30문항 검색 {seconds[name]}s")

    # 방식끼리 질문별 차이와, 문서 단위 근거를 하나도 못 찾은 질문
    by = {n: {r["question_id"]: r for r in res["rows"]} for n, res in results.items()}
    ids = [q["id"] for q in questions if by["dense"][q["id"]]["hit"] is not None]
    cmp = {}
    for a, b in (("hybrid", "dense"), ("hybrid", "bm25")):
        better = [q for q in ids if (by[a][q]["recall"] or 0) > (by[b][q]["recall"] or 0)]
        worse = [q for q in ids if (by[a][q]["recall"] or 0) < (by[b][q]["recall"] or 0)]
        cmp[f"{a}_vs_{b}"] = {"better": better, "worse": worse, "same": len(ids) - len(better) - len(worse)}
        print(f"{a} vs {b}: Recall이 더 높은 질문 {len(better)}개, 낮은 질문 {len(worse)}개, 같음 {len(ids) - len(better) - len(worse)}개")
    misses = {n: [q for q in ids if by[n][q]["hit"] == 0] for n in results}
    print("문서 단위로 근거를 하나도 못 찾은 질문:", {n: v for n, v in misses.items()})
    # STEP 9에서 Dense가 근거 페이지를 가져오지 못해 보류된 4건
    four = ["F-DEV-17", "F-DEV-22", "F-DEV-24", "F-DEV-25"]
    print("STEP 9의 보류 4건 (문서 단위 Recall dense / bm25 / hybrid, 페이지 단위):")
    for q in four:
        print(f"  {q}: " + " | ".join(f"{n} {by[n][q]['recall']:.2f}/{by[n][q]['page_recall']:.2f}" for n in results))

    out.mkdir(parents=True)
    (out / "manifest.json").write_text(json.dumps({
        "experiment_id": EXPERIMENT_ID, "track": "development", "step": "5", "k": 5, "question_set": "development",
        "questions": len(questions), "methods": list(methods), "bm25_tokenizer": sparse.tokenizer_name,
        "hybrid": HybridConfig().__dict__, "purpose": "비교 기록(승자 선정 아님)",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({
        "methods": {n: {**r["summary"], "search_seconds_30_questions": seconds[n]} for n, r in results.items()},
        "per_question_comparison": cmp, "no_document_found": misses}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out / "per_question.jsonl").open("w", encoding="utf-8") as fh:
        for n, r in results.items():
            for row in r["rows"]:
                fh.write(json.dumps({"method": n, **row}, ensure_ascii=False) + "\n")
    print(f"저장: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
