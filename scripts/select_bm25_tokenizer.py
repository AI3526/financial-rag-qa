"""STEP 5: BM25 토큰화 후보 4개를 development 질문으로 비교한다 (06 §5.3, DECISIONS.md D-035).

사용: uv run python scripts/select_bm25_tokenizer.py

- BM25 단독 검색, Top-5, 지표는 `book_id` 단위 Recall@5(공식)·Hit@5. `(book_id, page)` 단위는 참고로만 기록한다.
- 선정 규칙은 실행 전에 D-035에 고정했다: Recall@5 → Hit@5 → 동점이면 kiwi_userdict, kiwi, char2gram, whitespace 순.
  사용자 사전 후보(kiwi_userdict)를 포함한 4개와 제외한 3개, 두 경우의 승자를 모두 계산해 보여 준다(최종 선택은 사용자).
- 질문 문자열만 검색에 쓰고 정답은 검색이 끝난 뒤 지표 계산에만 쓴다. 로컬에만 저장한다(Langfuse 미사용).
- 결과 폴더가 이미 있으면 덮어쓰지 않고 중단한다. 승자 동결은 사용자가 승인한다.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.retrieval_compare import evaluate_retriever, pick_by_rule  # noqa: E402
from src.retrieval.sparse import SparseRetriever, load_chunks  # noqa: E402
from src.retrieval.tokenize import TOKENIZER_NAMES, get_tokenizer  # noqa: E402

EXPERIMENT_ID = "dev-bm25-tokenizer-001"
EVAL_DIR = ROOT / "data" / "finance_basic" / "evaluation"
TIE_ORDER = ["kiwi_userdict", "kiwi", "char2gram", "whitespace"]
SAMPLES = ["해온솔루션의 2026년 상반기 영업이익은 96억원입니다.", "루미센서 E는 확장형 제품이며 리브필름 R은 재생 원료 30%를 사용한다."]


def main() -> int:
    out = ROOT / "results" / EXPERIMENT_ID
    if out.exists():
        print(f"이미 결과가 있습니다(덮어쓰지 않음): {out}. (중단)")
        return 1
    questions = json.loads((EVAL_DIR / "development_questions.json").read_text(encoding="utf-8"))
    answers = {a["id"]: a for a in json.loads((EVAL_DIR / "development.json").read_text(encoding="utf-8"))}
    chunks = load_chunks()
    results, timing = {}, {}
    for name in TOKENIZER_NAMES:
        t = time.perf_counter()
        retriever = SparseRetriever(chunks, name, k=5)
        build = time.perf_counter() - t
        results[name] = evaluate_retriever(lambda q, r=retriever: r.retrieve(q, k=5).chunks, questions, answers)
        timing[name] = {"index_seconds": round(build, 2)}
        s = results[name]["summary"]
        print(f"{name:14} Hit@5={s['hit_at_5']:.4f} Recall@5={s['recall_at_5']:.4f} | 페이지 단위 Hit={s['page_hit_at_5']:.4f} "
              f"Recall={s['page_recall_at_5']:.4f} | 빈 결과 {s['empty_results']}개 | 색인 {build:.2f}s")
    summaries = {n: r["summary"] for n, r in results.items()}
    winner, reason = pick_by_rule(summaries, TIE_ORDER)
    no_dict = {n: v for n, v in summaries.items() if n != "kiwi_userdict"}
    winner_nd, reason_nd = pick_by_rule(no_dict, [n for n in TIE_ORDER if n != "kiwi_userdict"])
    print(f"규칙 적용 결과(4개 모두): 승자 {winner} (결정 기준: {reason})")
    print(f"규칙 적용 결과(사용자 사전 제외 3개): 승자 {winner_nd} (결정 기준: {reason_nd})")
    out.mkdir(parents=True)
    (out / "manifest.json").write_text(json.dumps({
        "experiment_id": EXPERIMENT_ID, "track": "development", "step": "5", "retrieval": "bm25", "k": 5,
        "question_set": "development", "questions": len(questions), "candidates": list(TOKENIZER_NAMES),
        "selection_rule": "DECISIONS.md D-035", "tie_order": TIE_ORDER, "rank_bm25": version("rank-bm25"),
        "kiwipiepy": version("kiwipiepy"), "chunking_version": chunks[0]["chunking_version"],
        "tokenizer_samples": {n: [get_tokenizer(n)(s) for s in SAMPLES] for n in TOKENIZER_NAMES},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({
        "models": {n: {**r["summary"], **timing[n]} for n, r in results.items()},
        "rule_result": {"winner_all_four": winner, "decided_by": reason,
                        "winner_without_userdict": winner_nd, "decided_by_without_userdict": reason_nd,
                        "frozen": False}}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    with (out / "per_question.jsonl").open("w", encoding="utf-8") as fh:
        for n, r in results.items():
            for row in r["rows"]:
                fh.write(json.dumps({"tokenizer": n, **row}, ensure_ascii=False) + "\n")
    print(f"저장: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
