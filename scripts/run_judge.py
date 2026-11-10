"""STEP 11: 저장된 서비스 응답을 LLM-as-a-Judge 4항목(사실, 조건, 근거, 판단)으로 채점한다 (D-040).

실행:
    uv run python scripts/run_judge.py --source dev-prompt-answer-v2-001 --out judge-dev-001 [--limit N] [--langfuse]

- 입력: `results/{source}/runs.jsonl`(질문, 답변과 상태, 인용, **실제 최종 문맥**)과 평가 파일의 참고 정답·필수 사실·금지 주장·
  기대 동작·채점 설명. 이 정보는 이 평가 단계에서만 쓰며 서비스 입력에는 들어가지 않는다. evidence_groups는 입력하지 않는다.
- `run_type`이 general_eval이 아닌 기록(모의 오류 등)은 일반 질문 평균에서 빼고 개수만 센다. 평가 파일에 없는 질문 ID도 센다.
- 서비스 오류(`error`) 응답은 모델을 부르지 않고 decision 0, 나머지 평가 불가로 기록한다.
- 채점 호출 실패와 출력 형식 오류는 평가 불가(0이나 1로 바꾸지 않음). 평균은 점수가 있는 답변만으로 구하고 개수를 함께 적는다.
- `--langfuse`: 채점 호출을 `judge-evaluation`(run_type:scoring) Trace로 기록한다. 정답 정보는 Langfuse에 보내지 않는다.
- 결과 폴더가 이미 있으면 덮어쓰지 않고 중단한다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_env  # noqa: E402

load_env()

from src.evaluation.judge import (ITEMS, aggregate, build_judge_llm, judge_one, load_judge_prompt,  # noqa: E402
                                  load_references)
from src.evaluation.scoring_trace import record_scoring  # noqa: E402


def load_targets(source: Path, refs: dict, limit: int | None) -> tuple[list[tuple[dict, dict]], dict]:
    records = [json.loads(line) for line in (source / "runs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    targets, skipped = [], {"not_general_eval": 0, "no_reference": 0}
    for r in records:
        if r.get("run_type") != "general_eval":
            skipped["not_general_eval"] += 1
        elif r.get("question_id") not in refs:
            skipped["no_reference"] += 1
        else:
            targets.append((r, refs[r["question_id"]]))
    return (targets[:limit] if limit else targets), skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--langfuse", action="store_true")
    ap.add_argument("--max-output-tokens", type=int, default=4096)
    args = ap.parse_args()

    source, out = ROOT / "results" / args.source, ROOT / "results" / args.out
    if not (source / "runs.jsonl").exists():
        print(f"입력이 없습니다: {source / 'runs.jsonl'}. (중단)")
        return 1
    if out.exists():
        print(f"이미 결과가 있습니다(덮어쓰지 않음): {out}. (중단)")
        return 1
    refs = load_references()
    targets, skipped = load_targets(source, refs, args.limit)
    spec = load_judge_prompt()
    model = os.environ["OPENAI_MODEL"]
    llm = build_judge_llm(model, max_completion_tokens=args.max_output_tokens)
    lf = None
    if args.langfuse:
        from src.observability.langfuse_client import get_langfuse

        lf = get_langfuse()
    print(f"채점 대상 {len(targets)}개 (일반 평가 아님 제외 {skipped['not_general_eval']}, 참고 정답 없음 제외 {skipped['no_reference']})")

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(judge_one, llm, spec, rec, ref): rec for rec, ref in targets}
        for n, fut in enumerate(as_completed(futures), start=1):
            rec, row = futures[fut], fut.result()
            if lf is not None and row["usage"] is not None:  # 모델을 부르지 않은 서비스 오류 행은 기록할 호출이 없다
                row["scoring_trace_id"] = record_scoring(
                    lf, name="judge-evaluation", model=model, track=rec.get("track") or "development", rag_method=row["rag_method"],
                    evaluated=rec, input={"question": rec["question"], "answer": rec["response"].get("answer"),
                                          "service_status": row["service_status"], "reference_data": "평가 단계 입력(Langfuse에는 보내지 않음)"},
                    output={"items": {k: {f: v[f] for f in ("status", "score", "reason")} for k, v in row["items"].items()}},
                    usage=row["usage"], metadata={"evaluator": "llm-judge", "prompt_version": spec["version"], "status_match": row["status_match"]})
            rows.append(row)
            s = {k: (row["items"][k]["score"] if row["items"][k]["status"] == "scored" else row["items"][k]["status"][:3]) for k in ITEMS}
            print(f"  [{n}/{len(targets)}] {str(row['question_id']):9} {row['rag_method']:13} {row['service_status']:19} {s} ({row['seconds']}초)")
    if lf is not None:
        lf.flush()
    rows.sort(key=lambda r: (str(r["question_id"]), r["rag_method"]))

    summary = aggregate(rows)
    usage_total = {k: sum((r.get("usage") or {}).get(k) or 0 for r in rows)
                   for k in ("calls", "input_tokens", "output_tokens", "reasoning_tokens", "cached_input_tokens")}
    out.mkdir(parents=True)
    with (out / "judge.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps({"by_method": summary, "skipped": skipped, "evaluation_usage": usage_total},
                                                 ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "out": args.out, "source": args.source, "metric": "llm-judge-4items", "model": model,
        "prompt": {"version": spec["version"], "sha256": spec["sha256"]}, "max_output_tokens": args.max_output_tokens,
        "reasoning_effort": "low", "concurrency": args.concurrency, "passes": 1, "answers": len(targets), "langfuse": bool(args.langfuse),
        "reference_data": "development/additional/expansion_validation 평가 파일(evidence_groups 제외)",
        "note": "평가 호출 사용량은 서비스 답변 생성 사용량과 분리해 기록한다. 해당 없음과 평가 불가는 평균에서 빼고 개수로 센다. "
                "상태 비교와 서비스 오류 처리는 코드가 한다.",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n요약:", json.dumps(summary, ensure_ascii=False))
    print("평가 호출 사용량:", usage_total)
    print(f"저장: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
