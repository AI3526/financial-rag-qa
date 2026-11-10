"""STEP 11: 저장된 서비스 응답을 LLM-as-a-Judge 4항목(사실, 조건, 근거, 판단)으로 채점한다 (D-040).

실행:
    uv run python scripts/run_judge.py --source dev-prompt-answer-v2-001 --out judge-v2-dev-001 [--prompt judge_v1|judge_v2] [--limit N] [--langfuse]

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

from src.evaluation.judge import (DEFAULT_PROMPT, ITEMS, aggregate_passes, build_judge_llm, combine_passes,  # noqa: E402
                                  judge_one, load_judge_prompt, load_references, response_format_for)
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
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="prompts/{이름}.json (judge_v1 또는 judge_v2, 기본은 최신 승인 문안)")
    ap.add_argument("--passes", type=int, default=1, help="채점 회차 수. 개발은 1회, 공식 실행은 3회(다수결 + 회차별 병기, D-044·D-045)")
    args = ap.parse_args()
    if args.passes < 1:
        print("--passes는 1 이상이어야 합니다. (중단)")
        return 1

    source, out = ROOT / "results" / args.source, ROOT / "results" / args.out
    if not (source / "runs.jsonl").exists():
        print(f"입력이 없습니다: {source / 'runs.jsonl'}. (중단)")
        return 1
    if out.exists():
        print(f"이미 결과가 있습니다(덮어쓰지 않음): {out}. (중단)")
        return 1
    refs = load_references()
    targets, skipped = load_targets(source, refs, args.limit)
    spec = load_judge_prompt(args.prompt)
    model = os.environ["OPENAI_MODEL"]
    llm = build_judge_llm(model, max_completion_tokens=args.max_output_tokens, response_format=response_format_for(spec))
    print(f"프롬프트 {spec['version']} (sha256 {spec['sha256'][:12]}…), 진단 항목: {spec.get('diagnostic') or '없음'}")
    lf = None
    if args.langfuse:
        from src.observability.langfuse_client import get_langfuse

        lf = get_langfuse()
    print(f"채점 대상 {len(targets)}개 (일반 평가 아님 제외 {skipped['not_general_eval']}, 참고 정답 없음 제외 {skipped['no_reference']})")

    rows: list[dict] = []
    total = len(targets) * args.passes
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        # 같은 답변을 회차마다 독립적으로 채점한다(회차 사이에 서로의 결과를 보지 않는다)
        futures = {pool.submit(judge_one, llm, spec, rec, ref): (rec, k) for rec, ref in targets for k in range(1, args.passes + 1)}
        for n, fut in enumerate(as_completed(futures), start=1):
            (rec, pass_no), row = futures[fut], fut.result()
            row["pass"] = pass_no
            if lf is not None and row["usage"] is not None:  # 모델을 부르지 않은 서비스 오류 행은 기록할 호출이 없다
                row["scoring_trace_id"] = record_scoring(
                    lf, name="judge-evaluation", model=model, track=rec.get("track") or "development", rag_method=row["rag_method"],
                    evaluated=rec, input={"question": rec["question"], "answer": rec["response"].get("answer"),
                                          "service_status": row["service_status"], "reference_data": "평가 단계 입력(Langfuse에는 보내지 않음)"},
                    output={"items": {k: {f: v[f] for f in ("status", "score", "reason")} for k, v in row["items"].items()},
                            "context_check": ({"status": row["context_check"]["status"], "coverage": row["context_check"]["coverage"],
                                               "facts": [{"index": f["index"], "in_context": f["in_context"]} for f in row["context_check"]["facts"]]}
                                              if row.get("context_check") else None)},
                    usage=row["usage"], metadata={"evaluator": "llm-judge", "prompt_version": spec["version"], "status_match": row["status_match"],
                                                  "pass": pass_no})
            rows.append(row)
            s = {k: (row["items"][k]["score"] if row["items"][k]["status"] == "scored" else row["items"][k]["status"][:3]) for k in ITEMS}
            cov = (row.get("context_check") or {}).get("coverage") or (row.get("context_check") or {}).get("status", "-")
            print(f"  [{n}/{total}] {str(row['question_id']):9} {row['rag_method']:13} 회차 {pass_no} {row['service_status']:19} {s} 문맥={cov} ({row['seconds']}초)")
    if lf is not None:
        lf.flush()
    rows.sort(key=lambda r: (str(r["question_id"]), r["rag_method"], r["pass"]))
    majority_rows = combine_passes(rows)  # 요청별 다수결. 회차가 1이면 그 결과와 같다

    stats = aggregate_passes(rows, majority_rows)
    summary = stats["majority"]
    usage_total = {k: sum((r.get("usage") or {}).get(k) or 0 for r in rows)
                   for k in ("calls", "input_tokens", "output_tokens", "reasoning_tokens", "cached_input_tokens")}
    out.mkdir(parents=True)
    with (out / "judge.jsonl").open("w", encoding="utf-8") as fh:  # (요청, 회차)마다 한 줄
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "judge_majority.jsonl").open("w", encoding="utf-8") as fh:  # 요청마다 한 줄(다수결, 회차별 votes 포함)
        for r in majority_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps({
        "by_method": summary, "passes": stats["passes"], "per_pass": stats["per_pass"], "range": stats["range"],
        "disagreement": stats["disagreement"], "ties": stats["ties"], "skipped": skipped, "evaluation_usage": usage_total},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "out": args.out, "source": args.source, "metric": "llm-judge-4items", "model": model,
        "prompt": {"version": spec["version"], "sha256": spec["sha256"], "diagnostic": spec.get("diagnostic")},
        "max_output_tokens": args.max_output_tokens,
        "reasoning_effort": "low", "concurrency": args.concurrency, "passes": args.passes, "answers": len(targets), "langfuse": bool(args.langfuse),
        "aggregation": ("회차 1회" if args.passes == 1 else "다수결(상태·점수 모두 절반 초과, 동점은 평가 불가) + 회차별 충족률·범위·갈린 답변 수 병기"),
        "reference_data": "development/additional/expansion_validation 평가 파일(evidence_groups 제외)",
        "note": "평가 호출 사용량은 서비스 답변 생성 사용량과 분리해 기록한다. 해당 없음과 평가 불가는 평균에서 빼고 개수로 센다. "
                "상태 비교와 서비스 오류 처리는 코드가 한다.",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n요약(다수결):", json.dumps(summary, ensure_ascii=False))
    if args.passes > 1:
        print("회차 사이 충족률 범위:", json.dumps(stats["range"], ensure_ascii=False))
        print("판정이 갈린 답변 수:", json.dumps(stats["disagreement"], ensure_ascii=False), "| 동점:", json.dumps(stats["ties"], ensure_ascii=False))
    print("평가 호출 사용량:", usage_total)
    print(f"저장: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
