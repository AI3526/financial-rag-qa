"""[확장] 공식 실행(official-001)의 응답시간 분위수(P50·P90·P95·P99)를 세 RAG 방식별로 계산한다 (읽기 전용, 새 호출 없음).

10_ADDITIONAL_METRICS §4.1~4.3 기준: 전체 API 응답시간(`performance.total_duration_ms`)으로 계산하고 정상 처리 건만 쓴다(실패 요청은 이번 실행에 없음).
분위수는 선형 보간(NumPy 기본 `linear`)이다. 요청이 60건뿐이라 P95·P99는 상위 몇 건의 값에 좌우돼 통계적으로 불안정하다(문서의 주의사항).
출력: results/official-001-ops/latency_percentiles.json·latency_percentiles.md (이미 있으면 덮어쓰지 않고 중단). 필수 운영 지표(operational.md)와 분리한 확장 지표다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = "official-001"
METHODS = ("dense", "hybrid", "hybrid_rerank")
QS = (50, 90, 95, 99)
FIELDS = (("total_duration_ms", "전체 응답시간"), ("retrieval_duration_ms", "검색 시간(재정렬 포함)"), ("generation_duration_ms", "생성 시간"))


def main() -> int:
    out = ROOT / "results" / f"{EXP}-ops"
    jp, mp = out / "latency_percentiles.json", out / "latency_percentiles.md"
    if jp.exists() or mp.exists():
        print("이미 있습니다(덮어쓰지 않음). (중단)")
        return 1
    runs = [json.loads(l) for l in (ROOT / "results" / EXP / "runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    runs = [r for r in runs if r.get("run_type") == "general_eval" and (r.get("response") or {}).get("status") != "error"]
    res = {}
    for m in METHODS:
        rs = [r for r in runs if r["rag_method"] == m]
        res[m] = {"n": len(rs)}
        for key, _ in FIELDS:
            xs = np.array([r["performance"][key] for r in rs if r["performance"].get(key) is not None], dtype=float)
            res[m][key] = {"n": int(len(xs)), "mean": float(xs.mean()), **{f"p{q}": float(np.percentile(xs, q)) for q in QS}, "max": float(xs.max())}
    jp.write_text(json.dumps({"experiment": EXP, "method": "numpy percentile linear", "scope": "general_eval 정상 처리 건", "result": res}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    f0 = lambda x: f"{x:,.0f}"  # noqa: E731
    L = [f"# [확장] 응답시간 분위수 — 공식 실행 `{EXP}`", "",
         "> 일반 평가 요청 중 정상 처리 건(방식당 60건)의 전체 API 응답시간으로 계산했다. 분위수는 선형 보간이다. **요청이 60건뿐이라 P95·P99는 가장 느린 몇 건에 좌우돼 불안정하다**(10_ADDITIONAL_METRICS §4.3 주의). 동시성 1 실행이다. 필수 운영 지표는 `operational.md`에 있다.", ""]
    for key, label in FIELDS:
        L += [f"## {label} (ms)", "", "| 방식 | 건수 | 평균 | P50 | P90 | P95 | P99 | 최대 |", "|---|---|---|---|---|---|---|---|"]
        for m in METHODS:
            d = res[m][key]
            L.append(f"| {m} | {d['n']} | {f0(d['mean'])} | {f0(d['p50'])} | {f0(d['p90'])} | {f0(d['p95'])} | {f0(d['p99'])} | {f0(d['max'])} |")
        L.append("")
    L += ["- 평균이 비슷해도 꼬리(P95 이상)는 방식마다 다를 수 있다. 위 표의 P95·P99는 참고용이고 방식 간 차이를 단정하는 근거로 쓰지 않는다.", ""]
    mp.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
