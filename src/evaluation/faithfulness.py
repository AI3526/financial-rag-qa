"""RAGAS Faithfulness 결과 처리 (STEP 11, 02_EVALUATION_PLAN.md §6.1, 과제 04_step6).

이 모듈은 ragas 없이 동작하는 순수 함수만 담는다(메인 환경에서 테스트). ragas를 부르는 코드는
`ragas_faithfulness.py`에 있고 격리 환경(`uv run --with ragas …`)에서만 실행한다(D-039).

기록 규칙(과제 기준):
- 점수 = 문맥이 뒷받침한 사실 주장 수 ÷ 사실 주장 수 (0~1).
- 사실 주장이 하나도 없으면 N/A(`not_applicable`)이고 0점이 아니다.
- 평가 호출이 실패하면 평가 불가(`unevaluable`)이며 0이나 1로 바꾸지 않는다.
- 평균은 점수가 있는 것만으로 내고, 평균 옆에 대상 수·N/A·평가 불가 수를 함께 적는다.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "faithfulness_statement_v1.json"
NA_REASON = "사실 주장 없음(되묻기·요청·확인 불가 안내만 있는 답변)"


def load_statement_prompt(path: Path = PROMPT_PATH) -> dict:
    """1단계(주장 쪼개기) 프롬프트 파일을 읽는다. 파일의 sha256과 버전을 함께 돌려준다."""
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    return {"version": data["version"], "sha256": hashlib.sha256(raw).hexdigest(),
            "instruction": data["instruction"], "examples": data["examples"]}


def build_detail(statements: list[str], verdicts: list[dict]) -> dict:
    """1단계 주장 목록과 2단계 판정(statement, reason, verdict)으로 결과를 만든다.

    - 주장이 없으면 N/A(점수 None). 주장은 있는데 판정이 없으면 평가 불가.
    - 점수는 verdict가 1인 판정 수 ÷ 판정 수다(RAGAS 자체 계산과 같은 방식).
    """
    if not statements:
        return {"status": "not_applicable", "score": None, "supported": 0, "total": 0, "statements": [], "reason": NA_REASON}
    if not verdicts:
        return {"status": "unevaluable", "score": None, "supported": 0, "total": 0,
                "statements": [{"statement": s, "verdict": None, "reason": None} for s in statements],
                "reason": "판정 결과 없음(2단계 응답이 비어 있음)"}
    items = [{"statement": v.get("statement"), "verdict": 1 if v.get("verdict") else 0, "reason": v.get("reason")} for v in verdicts]
    supported = sum(i["verdict"] for i in items)
    unsupported = [i["statement"] for i in items if not i["verdict"]]
    return {"status": "scored", "score": supported / len(items), "supported": supported, "total": len(items), "statements": items,
            "reason": f"사실 주장 {len(items)}개 중 {supported}개가 문맥으로 뒷받침됨" + (f". 뒷받침되지 않음: {unsupported}" if unsupported else "")}


def failed_result(exc: BaseException) -> dict:
    """평가 호출이 실패한 결과. 점수를 0이나 1로 바꾸지 않는다. 오류 문구는 종류와 짧은 설명만 남긴다."""
    return {"status": "unevaluable", "score": None, "supported": 0, "total": 0, "statements": [],
            "reason": f"평가 호출 실패: {type(exc).__name__}: {str(exc)[:160]}"}


def aggregate(rows: list[dict]) -> dict:
    """채점 결과 목록(요청 1건 × 횟수 N)을 방식별로 요약한다.

    rows의 각 원소: {"request_id", "rag_method", "pass", "status", "score"}.
    요청마다 점수가 있는 횟수의 평균과 범위를 구하고, 그 평균들의 평균을 방식별 점수로 쓴다.
    N/A와 평가 불가는 평균에서 빼고 개수로 따로 센다.
    """
    by_method: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        by_method.setdefault(r["rag_method"], {}).setdefault(r["request_id"], []).append(r)
    out: dict[str, Any] = {}
    for method, reqs in by_method.items():
        means, ranges, na, unevaluable, scored_requests = [], [], 0, 0, 0
        for runs in reqs.values():
            scores = [r["score"] for r in runs if r["status"] == "scored"]
            if scores:
                scored_requests += 1
                means.append(statistics.mean(scores))
                ranges.append(max(scores) - min(scores))
            elif any(r["status"] == "unevaluable" for r in runs):
                unevaluable += 1
            else:
                na += 1
        out[method] = {
            "answers": len(reqs), "scored": scored_requests, "not_applicable": na, "unevaluable": unevaluable,
            "mean": statistics.mean(means) if means else None,
            "mean_range_across_passes": statistics.mean(ranges) if ranges else None,
            "max_range_across_passes": max(ranges) if ranges else None,
            "answers_with_pass_disagreement": sum(1 for x in ranges if x > 1e-9),
        }
    return out
