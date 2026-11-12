"""STEP 13: 보안 비교 요청 4건이 Langfuse에 기록돼 있는지 로컬 run 레코드와 대조한다 (Langfuse 읽기만 하고 쓰지 않는다).

사용: uv run python scripts/verify_security_langfuse.py [실험ID]   (기본 security-001)

- 검사는 STEP 10과 같은 `src/observability/trace_checks.check_trace`를 쓴다(루트 Trace, Trace·Request ID 연결, 필수 식별 정보, 모델 입출력·사용량, 비밀 값 미노출 등).
- 방어 프롬프트 실행은 저장한 검색 결과를 재사용해(재검색 없음) 검색 Span이 없으므로 검색 관련 검사가 실패할 수 있다. 그 경우 실패한 검사 이름을 그대로 보고하며 통과로 바꾸지 않는다.
- 결과는 `results/{실험ID}/langfuse_report.json`에 저장한다(이미 있으면 덮어쓰지 않음). 비밀 값은 출력하지 않는다.
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

from src.observability.langfuse_client import get_langfuse  # noqa: E402
from src.observability.trace_checks import check_trace  # noqa: E402


def main() -> int:
    experiment = sys.argv[1] if len(sys.argv) > 1 else "security-001"
    out = ROOT / "results" / experiment
    report_path = out / "langfuse_report.json"
    if report_path.exists():
        print(f"이미 있습니다(덮어쓰지 않음): {report_path}. (중단)")
        return 1
    lf = get_langfuse()
    if lf is None:
        print("Langfuse 설정을 읽지 못했습니다. (중단)")
        return 1
    records = [json.loads(l) for l in (out / "runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    result = []
    for rec in records:
        tid = rec.get("trace_id")
        obs = []
        for _ in range(24):  # 도착을 기다린다(최대 약 2분)
            obs = lf.api.observations.get_many(trace_id=tid, limit=100, fields="core,basic,io,model,usage,metadata").data
            if sum(o.type == "GENERATION" for o in obs) >= len(rec.get("llm_calls") or []) and any(o.name == "rag-query" and o.output for o in obs):
                break
            time.sleep(5)
        checks = check_trace(rec, obs)
        bad = [(n, d) for n, ok, d in checks if not ok]
        names = sorted({o.name for o in obs})
        print(f"\n{rec['question_id']} / {rec['versions'].get('prompt_version')} / 요청 {rec['request_id'][:8]}… Trace {str(tid)[:8]}… 관측 {len(obs)}개, 검사 {len(checks)}개 중 통과 {len(checks) - len(bad)}개")
        print("  관측 이름:", names)
        for n, d in bad:
            print(f"  실패: {n} — {d}")
        result.append({"request_id": rec["request_id"], "question_id": rec["question_id"], "prompt_version": rec["versions"].get("prompt_version"), "run_type": rec.get("run_type"),
                       "trace_id": tid, "observations": len(obs), "observation_names": names, "checks": [{"name": n, "passed": ok, "detail": d} for n, ok, d in checks],
                       "failed": [{"name": n, "detail": d} for n, d in bad], "trace_url": lf.get_trace_url(trace_id=tid) if tid else None})
    report = {"experiment_id": experiment, "step": "13", "requests": result, "all_passed": all(not r["failed"] for r in result),
              "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n전체 통과:", report["all_passed"], "| 저장:", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
