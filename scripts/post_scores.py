"""STEP 12: 최종 평가 값을 서비스 Trace의 Score로 Langfuse에 올린다 (DECISIONS D-045).

사용:
    uv run python scripts/post_scores.py --compare official-001-compare              # 계획만 출력(기본, 아무것도 쓰지 않음)
    uv run python scripts/post_scores.py --compare official-001-compare --apply      # 실제로 Langfuse에 기록

- 입력은 `compare_official.py`가 만든 `per_question.jsonl`(질문×방식별 **최종 값**). 값이 없는 항목(해당 없음·평가 불가)은 올리지 않는다(0으로 바꾸지 않음).
- Score는 요청 Trace(`trace_id`)에 붙고, 이름은 `{지표}`이며 `score_id`는 `{request_id}:{지표}`로 고정해 다시 실행해도 중복 기록되지 않는다(같은 ID는 갱신).
- 기본은 계획만 출력한다. `--apply`는 외부 서비스에 쓰므로 실행 전 건수를 확인한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_env  # noqa: E402

load_env()

from src.evaluation.official import METRICS  # noqa: E402


def plan(rows: list[dict]) -> list[dict]:
    """per_question 행에서 올릴 Score 목록을 만든다. 값이 None인 항목은 제외한다."""
    out = []
    for r in rows:
        if not r.get("trace_id"):
            continue
        for key, label in METRICS:
            v = r.get(key)
            if v is None:
                continue
            out.append({"trace_id": r["trace_id"], "name": key, "value": float(v), "score_id": f"{r['request_id']}:{key}",
                        "comment": f"{label} | {r['question_id']} | {r['rag_method']}"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", required=True, help="compare_official.py 결과 폴더 이름")
    ap.add_argument("--apply", action="store_true", help="실제로 Langfuse에 기록(없으면 계획만 출력)")
    args = ap.parse_args()
    path = ROOT / "results" / args.compare / "per_question.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    items = plan(rows)
    skipped = len(rows) * len(METRICS) - len(items)
    print(f"요청 {len(rows)}건, 올릴 Score {len(items)}개, 값이 없어 올리지 않는 항목 {skipped}개")
    if not args.apply:
        print("계획만 출력했습니다(--apply 없음). 아무것도 기록하지 않았습니다.")
        return 0
    from src.observability.langfuse_client import get_langfuse
    lf = get_langfuse()
    if lf is None:
        print("Langfuse 클라이언트를 만들 수 없습니다(키 확인). (중단)")
        return 1
    failed = 0
    for it in items:
        try:
            lf.create_score(name=it["name"], value=it["value"], trace_id=it["trace_id"], score_id=it["score_id"], data_type="NUMERIC", comment=it["comment"])
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"실패 {it['score_id']}: {type(e).__name__}")
    lf.flush()
    print(f"기록 요청 {len(items) - failed}개, 실패 {failed}개")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
