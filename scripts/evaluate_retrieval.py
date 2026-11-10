"""STEP 11: 서비스 실행 기록의 검색 결과로 Hit@k·Recall@k를 공식 형식으로 계산한다 (과제 04_step6 "검색 지표").

실행:
    uv run python scripts/evaluate_retrieval.py --source dev-prompt-answer-v2-001 dev-hybrid-answer-v2-001 \\
        dev-hybrid-rerank-answer-v2-001 --out retrieval-eval-dev-001

- 입력: `results/{source}/runs.jsonl`(서비스가 남긴 run 레코드의 검색 결과와 최종 문맥)과 평가 파일의 `evidence_groups`.
  모델 호출·Langfuse 기록이 없다(계산만 한다).
- k는 답변 생성 직전에 선택한 상위 청크 수이며(기록된 `final_k`) 세 방식에 같게 둔다. 문서 단위(`book_id`)로 중복을 제거해 센다.
- 참고 근거가 비어 있는 문항은 해당 없음(이유 기록), 서비스 오류로 검색 기록이 없으면 평가 불가, 검색 결과가 비어도 참고 문서가
  있으면 0으로 계산한다. 같은 참고 목록을 세 방식에 쓴다.
- 문맥 토큰 한도 이후의 최종 문맥으로도 같은 지표를 따로 낸다(필요한 근거가 실제 문맥에 남았는지, 과제 65행).
- 결과 폴더가 이미 있으면 덮어쓰지 않고 중단한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.judge import load_references  # noqa: E402
from src.evaluation.retrieval_metrics import aggregate_records, evaluate_record  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", nargs="+", required=True, help="results/ 아래 서비스 실험 폴더(여러 개 가능)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / "results" / args.out
    if out.exists():
        print(f"이미 결과가 있습니다(덮어쓰지 않음): {out}. (중단)")
        return 1
    refs = load_references()
    rows, skipped = [], {"not_general_eval": 0, "no_reference": 0}
    for source in args.source:
        path = ROOT / "results" / source / "runs.jsonl"
        if not path.exists():
            print(f"입력이 없습니다: {path}. (중단)")
            return 1
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("run_type") != "general_eval":
                skipped["not_general_eval"] += 1
            elif rec.get("question_id") not in refs:
                skipped["no_reference"] += 1
            else:
                ref = refs[rec["question_id"]]
                rows.append({"source": source, "request_id": rec["request_id"], "question_id": rec["question_id"],
                             "rag_method": rec["rag_method"], "evaluation_set_type": rec.get("evaluation_set_type"),
                             "expected_action": ref["expected_action"],
                             "metrics": evaluate_record(rec, ref["evidence_groups"], ref["expected_action"])})
    summary = aggregate_records(rows)
    ks = sorted({k for s in summary.values() for k in s["k"]})
    out.mkdir(parents=True)
    with (out / "retrieval_eval.jsonl").open("w", encoding="utf-8") as fh:
        for r in sorted(rows, key=lambda r: (r["question_id"], r["rag_method"])):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps({"by_method": summary, "k_values": ks, "skipped": skipped}, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "out": args.out, "sources": args.source, "metric": "hit@k, recall@k (book_id 단위, 공식) + (book_id, page) 단위 참고",
        "rows": len(rows), "k_values": ks, "llm_calls": 0, "langfuse": False,
        "note": "참고 근거 목록은 평가 파일의 evidence_groups이며 방식마다 바꾸지 않는다. 최종 문맥 기준 지표는 토큰 한도로 제외된 청크를 반영한다.",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for method, s in summary.items():
        fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
        print(f"{method:14} 문항 {s['questions']} 점수 {s['scored']} N/A {s['not_applicable']} 평가불가 {s['unevaluable']} k={s['k']} | "
              f"Hit {fmt(s['hit_at_k'])} Recall {fmt(s['recall_at_k'])} | 쪽 단위 Hit {fmt(s['page_hit_at_k'])} Recall {fmt(s['page_recall_at_k'])} | "
              f"최종 문맥 Recall {fmt(s['final_context_recall'])} (잘린 문항 {s['questions_with_dropped_chunks']})")
    print(f"저장: {out} (건너뜀 {skipped})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
