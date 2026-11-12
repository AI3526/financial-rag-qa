"""STEP 14: 공식 실행(official-001)의 운영 지표를 세 RAG 방식별로 정리한다 (읽기 전용, 새 모델·검색 호출 없음).

필수 지표만 보고한다(08 §18, 12 §5): 정상 처리 건의 평균 응답시간과 건수, 오류율과 실패 건수, 입력·출력 토큰 합계와 사용량을 확인한 요청 수, 비용(단가 확인 가능 시)과 확인한 요청 수.
평균 응답시간의 구간(검색, 재정렬, 생성), 재시도, 응답 상태 분포, 질문별 시간·토큰 표, 평가 호출 사용량(서비스 생성과 분리)을 함께 적는다.
최소·중앙·최대 시간은 설명용 참고이며 P95·P99, 동시성, 처리량은 [확장]이라 계산하지 않는다(10_ADDITIONAL_METRICS).

입력: results/official-001/runs.jsonl(일반 평가 180건), official-001-judge·-faithfulness의 evaluation_usage, official-001/experiment_log.json.
출력: results/official-001-ops/{operational.json, operational.md, per_question_ops.csv, manifest.json} (이미 있으면 덮어쓰지 않고 중단)

사용: uv run python scripts/ops_summary_official.py
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.compare import operational  # noqa: E402

EXP = "official-001"
METHODS = ("dense", "hybrid", "hybrid_rerank")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def tokens(rec: dict, key: str) -> int | None:
    calls = rec.get("llm_calls") or []
    if not calls or any(c.get(key) is None for c in calls):
        return None
    return sum(c[key] for c in calls)


def main() -> int:
    out = ROOT / "results" / f"{EXP}-ops"
    if out.exists():
        print(f"이미 결과가 있습니다(덮어쓰지 않음): {out}. (중단)")
        return 1
    runs = [r for r in read_jsonl(ROOT / "results" / EXP / "runs.jsonl") if r.get("run_type") == "general_eval"]
    log = json.loads((ROOT / "results" / EXP / "experiment_log.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((ROOT / "results" / EXP / "manifest.json").read_text(encoding="utf-8"))
    env, ps = run_manifest["environment"], run_manifest["snapshot"]["pipeline_settings"]
    judge = json.loads((ROOT / "results" / f"{EXP}-judge" / "summary.json").read_text(encoding="utf-8"))["evaluation_usage"]
    faith = json.loads((ROOT / "results" / f"{EXP}-faithfulness" / "summary.json").read_text(encoding="utf-8"))["evaluation_usage"]
    by = {m: [r for r in runs if r["rag_method"] == m] for m in METHODS}
    ops = {m: operational(rs) for m, rs in by.items()}

    def stat(m: str, key: str) -> dict:
        xs = [r["performance"][key] for r in by[m] if (r.get("response") or {}).get("status") != "error" and r["performance"].get(key) is not None]
        return {"n": len(xs), "mean": statistics.mean(xs), "median": statistics.median(xs), "min": min(xs), "max": max(xs)} if xs else {"n": 0}

    extra = {}
    for m in METHODS:
        rs = by[m]
        ok = [r for r in rs if (r.get("response") or {}).get("status") != "error"]
        rerank = [r["performance"].get("reranking_duration_ms") for r in ok if r["performance"].get("reranking_duration_ms") is not None]
        extra[m] = {
            "total_ms": stat(m, "total_duration_ms"),
            "status_counts": dict(Counter((r.get("response") or {}).get("status") for r in rs)),
            "mean_reranking_ms": statistics.mean(rerank) if rerank else None, "reranking_known_requests": len(rerank),
            "reasoning_tokens": sum(c.get("reasoning_tokens") or 0 for r in rs for c in (r.get("llm_calls") or [])),
            "cached_input_tokens": sum(c.get("cached_input_tokens") or 0 for r in rs for c in (r.get("llm_calls") or [])),
            "mean_input_tokens_per_request": ops[m]["input_tokens"] / ops[m]["usage_known_requests"] if ops[m]["usage_known_requests"] else None,
            "mean_output_tokens_per_request": ops[m]["output_tokens"] / ops[m]["usage_known_requests"] if ops[m]["usage_known_requests"] else None,
            "mean_cost_per_request_usd": ops[m]["cost_usd"] / ops[m]["cost_known_requests"] if ops[m]["cost_known_requests"] else None,
        }
    # 질문별 표
    qids = sorted({r["question_id"] for r in runs})
    rows = []
    for q in qids:
        row = {"question_id": q}
        for m in METHODS:
            rec = next((r for r in by[m] if r["question_id"] == q), None)
            row[f"{m}_status"] = (rec.get("response") or {}).get("status") if rec else None
            row[f"{m}_total_ms"] = rec["performance"].get("total_duration_ms") if rec else None
            row[f"{m}_input_tokens"] = tokens(rec, "input_tokens") if rec else None
            row[f"{m}_output_tokens"] = tokens(rec, "output_tokens") if rec else None
        rows.append(row)
    # 느려진 질문 수(hybrid_rerank가 hybrid보다 1초 넘게 느린 질문)
    slower = sum(1 for r in rows if r["hybrid_rerank_total_ms"] and r["hybrid_total_ms"] and r["hybrid_rerank_total_ms"] - r["hybrid_total_ms"] > 1000)
    result = {"experiment": EXP, "requests_per_method": {m: len(by[m]) for m in METHODS}, "operational": ops, "extra": extra,
              "run_wall_clock": {"started_at": log.get("started_at"), "finished_at": log.get("finished_at")},
              "evaluation_usage_separate": {"judge": judge, "faithfulness": faith}, "hybrid_rerank_slower_than_hybrid_by_1s_questions": slower}
    out.mkdir(parents=True)
    (out / "operational.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out / "per_question_ops.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    f0 = lambda x: "-" if x is None else f"{x:,.0f}"  # noqa: E731
    f4 = lambda x: "-" if x is None else f"{x:.4f}"  # noqa: E731
    L = [f"# STEP 14 운영 지표 — 공식 실행 `{EXP}`", "",
         "> 대상: 일반 평가 요청(`run_type=general_eval`) 질문 60개 × 3방식 = 180건. 평가 호출(Judge, Faithfulness), 보안 질문, 모의 오류 요청, 예행(warmup)·중단된 실행(`official-001-aborted-*`)은 제외했다. 요청을 한 번에 하나씩 보냈다(동시성 1).",
         "> 응답시간은 API 요청 수신부터 최종 응답 반환까지이며 검색, 재정렬, 생성, 재시도를 포함한다. Qdrant Cloud·Langfuse Cloud와의 네트워크 시간이 포함돼 있다. 사용량·비용을 확인하지 못한 요청은 0이 아니라 확인 불가로 센다(이번에는 모두 확인됨).", "",
         "## 1. 필수 운영 지표", "", "| 지표 | dense | hybrid | hybrid_rerank |", "|---|---|---|---|"]
    row = lambda name, fn: L.append(f"| {name} | " + " | ".join(fn(m) for m in METHODS) + " |")  # noqa: E731
    row("요청 수", lambda m: str(ops[m]["requests"]))
    row("정상 처리 건수", lambda m: str(ops[m]["ok"]))
    row("시스템 오류(실패) 건수", lambda m: str(ops[m]["errors"]))
    row("오류율", lambda m: f"{ops[m]['error_rate']:.3f}")
    row("평균 응답시간(ms, 정상 건)", lambda m: f0(ops[m]["mean_total_ms"]))
    row("입력 토큰 합계(사용량 확인한 요청)", lambda m: f"{ops[m]['input_tokens']:,}")
    row("출력 토큰 합계(사용량 확인한 요청)", lambda m: f"{ops[m]['output_tokens']:,}")
    row("사용량 확인 요청 / 확인 불가", lambda m: f"{ops[m]['usage_known_requests']} / {ops[m]['usage_unknown_requests']}")
    row("비용 합계(USD, 단가 확인 요청)", lambda m: f4(ops[m]["cost_usd"]))
    row("비용 확인 요청 / 확인 불가", lambda m: f"{ops[m]['cost_known_requests']} / {ops[m]['cost_unknown_requests']}")
    L += ["", "- 오류율은 재시도 후에도 시스템 오류(`status=error`)로 끝난 일반 평가 요청 기준이다. `needs_clarification`·`abstained`는 시스템 오류가 아니다.",
          "- 비용 단가: 요청 기록의 `price_source`(OpenAI 모델 문서, 2026-11-09 확인)와 `configs/pricing.json`에 따른다.", "",
          "## 2. 시간 구간과 사용량 상세(참고)", "", "| 항목 | dense | hybrid | hybrid_rerank |", "|---|---|---|---|"]
    row("평균 검색 시간(ms, 재정렬 포함)", lambda m: f0(ops[m]["mean_retrieval_ms"]))
    row("평균 재정렬 시간(ms)", lambda m: "해당 없음" if extra[m]["mean_reranking_ms"] is None else f0(extra[m]["mean_reranking_ms"]))
    row("평균 생성 시간(ms)", lambda m: f0(ops[m]["mean_generation_ms"]))
    row("재시도 수", lambda m: str(ops[m]["retries"]))
    row("응답시간 최소 / 중앙 / 최대(ms, 설명용)", lambda m: f"{f0(extra[m]['total_ms']['min'])} / {f0(extra[m]['total_ms']['median'])} / {f0(extra[m]['total_ms']['max'])}")
    row("요청당 평균 입력 / 출력 토큰", lambda m: f"{extra[m]['mean_input_tokens_per_request']:,.0f} / {extra[m]['mean_output_tokens_per_request']:,.0f}")
    row("캐시된 입력 토큰 합계(설명용)", lambda m: f"{extra[m]['cached_input_tokens']:,}")
    row("요청당 평균 비용(USD)", lambda m: f4(extra[m]["mean_cost_per_request_usd"]))
    row("응답 상태(answered / needs_clarification / abstained)", lambda m: " / ".join(str(extra[m]["status_counts"].get(s, 0)) for s in ("answered", "needs_clarification", "abstained")))
    L += ["", f"- hybrid_rerank가 hybrid보다 1초 넘게 느린 질문: {slower}개 / 60개 (질문별 표는 `per_question_ops.csv`).", "",
          "## 3. 평가 호출 사용량(서비스 답변 생성과 분리)", "", "| 평가 | 호출 수 | 입력 토큰 | 출력 토큰 | 추론 토큰 | 캐시된 입력 토큰 |", "|---|---|---|---|---|---|"]
    for name, u in (("Judge(3회 다수결)", judge), ("RAGAS Faithfulness(3회)", faith)):
        L.append(f"| {name} | {u['calls']:,} | {u['input_tokens']:,} | {u['output_tokens']:,} | {u['reasoning_tokens']:,} | {u['cached_input_tokens']:,} |")
    L += ["", "- 평가 호출은 서비스 응답시간·토큰·비용 집계에 포함하지 않았다. 평가 호출의 시간은 기록돼 있지 않다.", "",
          "## 4. 해석", "",
          "- 세 방식 모두 시스템 오류 0건, 재시도 0건이다. 오류율 차이는 없다.",
          f"- **응답시간:** hybrid_rerank의 평균이 {f0(ops['hybrid_rerank']['mean_total_ms'])}ms로 dense({f0(ops['dense']['mean_total_ms'])}ms)보다 약 {f0(ops['hybrid_rerank']['mean_total_ms'] - ops['dense']['mean_total_ms'])}ms, hybrid({f0(ops['hybrid']['mean_total_ms'])}ms)보다 약 {f0(ops['hybrid_rerank']['mean_total_ms'] - ops['hybrid']['mean_total_ms'])}ms 길다. 차이는 재정렬 시간(평균 {f0(extra['hybrid_rerank']['mean_reranking_ms'])}ms)에서 오고, 생성 시간은 세 방식이 비슷하다.",
          "- **토큰·비용:** 입력 토큰은 세 방식이 거의 같고(문맥 한도와 최종 청크 수를 같게 두었기 때문), 출력 토큰은 hybrid_rerank가 가장 적다. 총 비용은 세 방식 모두 약 0.0175~0.0177 USD(60건)로 사실상 같다.",
          "- 품질이 비슷하면 시간·토큰·오류율·복잡도를 본다는 선택 규칙(12 §4.3)에서 운영 지표만으로 hybrid_rerank를 배제할 만한 차이는 없지만 약 0.3~0.4초 더 걸린다.", "",
          "## 5. 한계", "",
          "- 질문 60개, 한 번 실행한 결과다. 같은 질문을 반복하면 응답시간과 생성 길이가 달라질 수 있다.",
          f"- 실행 환경: {env['platform']}, Python {env['python']}, GPU {env['gpu']}. 서비스 응답시간에는 Qdrant Cloud·Langfuse Cloud와의 네트워크 시간이 포함돼 있다(TASKS의 보고 규칙). 실행 환경이 다르면 절대값이 달라진다.",
          f"- 실행 설정(동결, 세 방식 동일): 최종 청크 수 {ps['final_k']}, 문맥 {ps['context_limit_tokens']}토큰, 모델 timeout {ps['model_timeout_seconds']:.0f}초, 최대 모델 호출 {ps['max_model_calls']}회, 추론 강도 {ps['reasoning_effort']}, 최대 출력 토큰 {ps['max_completion_tokens']}, git `{run_manifest['git_commit'][:7]}`.",
          "- P95·P99, 동시 요청, 처리량, 캐시 지표는 [확장]이며 이번에는 측정하지 않았다.", ""]
    (out / "operational.md").write_text("\n".join(L), encoding="utf-8")
    manifest = {"experiment": EXP, "source": [f"results/{EXP}/runs.jsonl", f"results/{EXP}-judge/summary.json", f"results/{EXP}-faithfulness/summary.json", f"results/{EXP}/experiment_log.json"],
                "scope": "general_eval 180건(60문항 x 3방식), 평가 호출·보안·모의 오류·예행·중단 실행 제외", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n".join(L[:60]))
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
