"""STEP 11: 저장된 서비스 응답을 RAGAS Faithfulness로 채점한다.

실행(ragas는 격리 환경에서만 쓴다, D-039):
    uv run --with ragas --with "langchain-community<0.4" python scripts/run_ragas.py \\
        --source dev-prompt-answer-v2-001 --out faithfulness-dev-001 [--passes 1] [--limit N] [--langfuse]

- 입력은 `results/{source}/runs.jsonl`(서비스가 남긴 run 레코드)뿐이다: 질문, 답변, **실제 최종 문맥**(검색 결과 전체가 아니라
  모델에 전달한 청크). 정답·근거 파일은 읽지 않는다(Faithfulness는 정답이 필요 없다).
- 서비스 오류 응답(error)은 답변이 없어 평가 대상이 아니며 개수만 따로 센다.
- 사실 주장이 없는 답변은 N/A, 평가 호출 실패는 평가 불가로 기록한다(0점으로 바꾸지 않는다).
- development는 1회(`--passes 1`), 공식 실행은 3회 평균(`--passes 3`, 각 회의 점수와 이유도 저장)을 쓴다.
- `--langfuse`를 주면 평가 호출을 `faithfulness-evaluation`(run_type:scoring)으로 Langfuse에 기록한다. development 실험은 로컬에만 남기는 것이
  기본이다(D-029).
- 결과 폴더가 이미 있으면 덮어쓰지 않고 중단한다.
"""
from __future__ import annotations

import os

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")  # ragas의 사용 통계 전송을 끈다(ragas를 import하기 전에)

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_env  # noqa: E402

load_env()

from src.evaluation.faithfulness import aggregate, load_statement_prompt  # noqa: E402
from src.evaluation.ragas_faithfulness import build_evaluator, evaluate_many, tool_versions  # noqa: E402
from src.evaluation.scoring_trace import record_scoring  # noqa: E402


def load_items(source: Path, limit: int | None) -> tuple[list[dict], dict]:
    records = [json.loads(line) for line in (source / "runs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    items, skipped = [], {"service_error": 0, "no_context": 0}
    for r in records:
        resp = r.get("response") or {}
        if resp.get("status") == "error" or not resp.get("answer"):
            skipped["service_error"] += 1
            continue
        contexts = [c["text"] for c in (r.get("final_context") or {}).get("chunks", [])]
        if not contexts:  # 최종 문맥이 비어 있으면 Faithfulness를 계산할 수 없다(입력 오류 방지). 평가 불가가 아니라 대상 제외로 센다
            skipped["no_context"] += 1
            continue
        items.append({"request_id": r["request_id"], "trace_id": r.get("trace_id"), "question_id": r.get("question_id"),
                      "rag_method": r["rag_method"], "question": r["question"], "answer": resp["answer"],
                      "service_status": resp["status"], "contexts": contexts, "track": r.get("track") or "development"})
    return (items[:limit] if limit else items), skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="results/ 아래 서비스 실험 폴더 이름(runs.jsonl이 있는 곳)")
    ap.add_argument("--out", required=True, help="결과를 저장할 results/ 아래 새 폴더 이름")
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--langfuse", action="store_true")
    ap.add_argument("--max-output-tokens", type=int, default=4096, help="평가 호출당 출력 한도(ragas 기본 1024는 추론 토큰 때문에 부족했다)")
    args = ap.parse_args()

    source, out = ROOT / "results" / args.source, ROOT / "results" / args.out
    if not (source / "runs.jsonl").exists():
        print(f"입력이 없습니다: {source / 'runs.jsonl'}. (중단)")
        return 1
    if out.exists():
        print(f"이미 결과가 있습니다(덮어쓰지 않음): {out}. (중단)")
        return 1
    items, skipped = load_items(source, args.limit)
    spec = load_statement_prompt()
    metric = build_evaluator(spec=spec, max_output_tokens=args.max_output_tokens)
    model = os.environ["OPENAI_MODEL"]
    lf = None
    if args.langfuse:
        from src.observability.langfuse_client import get_langfuse

        lf = get_langfuse()
    by_request = {i["request_id"]: i for i in items}
    done = {"n": 0}

    def on_done(item: dict, row: dict) -> None:
        done["n"] += 1
        if lf is not None:
            row["scoring_trace_id"] = record_scoring(lf, name="faithfulness-evaluation", model=model, track=item["track"], rag_method=item["rag_method"],
                           evaluated=item, input={"question": item["question"], "answer": item["answer"], "context_count": len(item["contexts"])},
                           output={k: row.get(k) for k in ("status", "score", "reason", "statements")}, usage=row.get("usage"),
                           metadata={"evaluator": "ragas-faithfulness", "ragas_version": tool_versions()["ragas"],
                                     "prompt_version": spec["version"], "pass": row["pass"]})
        print(f"  [{done['n']}/{len(items) * args.passes}] {str(item['question_id']):9} {item['rag_method']:13} 회차 {row['pass']} "
              f"{row['status']:14} 점수 {row['score'] if row['score'] is None else round(row['score'], 3)} ({row['seconds']}초)")

    print(f"평가 대상 답변 {len(items)}개 × {args.passes}회 (서비스 오류 제외 {skipped['service_error']}, 문맥 없음 제외 {skipped['no_context']})")
    rows = asyncio.run(evaluate_many(metric, items, args.passes, args.concurrency, on_done))
    if lf is not None:
        lf.flush()

    summary = aggregate(rows)
    usage_total = {k: sum((r.get("usage") or {}).get(k, 0) for r in rows) for k in
                   ("calls", "input_tokens", "output_tokens", "reasoning_tokens", "cached_input_tokens")}
    out.mkdir(parents=True)
    with (out / "faithfulness.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({**r, "question": by_request[r["request_id"]]["question"]}, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps({"by_method": summary, "skipped": skipped, "evaluation_usage": usage_total},
                                                 ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "out": args.out, "source": args.source, "metric": "ragas-faithfulness", "tools": tool_versions(), "model": model,
        "statement_prompt": {"version": spec["version"], "sha256": spec["sha256"]},
        "nli_prompt": "ragas 0.4.3 기본값(수정 없음)", "max_output_tokens": args.max_output_tokens, "passes": args.passes, "concurrency": args.concurrency,
        "answers": len(items), "langfuse": bool(args.langfuse), "track": "development",
        "note": "평가 호출의 사용량은 서비스 답변 생성 사용량과 분리해 기록한다. N/A와 평가 불가는 평균에서 빼고 개수로 센다.",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print("\n요약:", json.dumps(summary, ensure_ascii=False))
    print("평가 호출 사용량:", usage_total)
    print(f"저장: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
