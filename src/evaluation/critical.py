"""STEP 12: 중대 오류 분류 입력·검증·사람 확인 대상 선정과 대표 사례 선정 (12_ACCEPTANCE_CRITERIA §5, DECISIONS D-045).

흐름(방식 A, 사용자 승인): 이 모듈이 Judge 0점 답변을 모아 분류 입력을 만들면, **이 세션의 Claude가 직접 읽고** C1~C5 해당 여부를 JSON으로 기록한다
(`critical_classification.jsonl`, `classified_by`에 분류 주체를 적는다). 이 모듈은 그 기록을 검증하고, 사람이 원문과 대조할 대상(후보 전부 + 방식별 무작위 표본)을 정한다.
- 모델 호출이 없다. 평가 정답은 분류 입력에 들어가도 되지만(평가 단계) 생성·검색 입력과는 무관하다.
- 선택에는 **사람이 확인한 건수**만 쓰고, AI만 분류한 건수는 따로 보고한다.
"""
from __future__ import annotations

import math
import random
from typing import Iterable

from src.evaluation.judge import format_citations, format_context, format_source_table
from src.evaluation.official import JUDGE_ITEMS, METHODS

CRITERIA = ("C1", "C2", "C3", "C4", "C5")
CRITERIA_LABEL = {"C1": "핵심 수치 오류", "C2": "근거 없는 사실·금지 주장", "C3": "대상 혼동", "C4": "부적절한 확정(또는 해야 할 보류·되묻기 누락)", "C5": "핵심 조건 위반"}
CAUSES = ("retrieval_failure", "context_truncation", "generation_error", "question_ambiguity", "evaluator_issue", "other", "none")
SAMPLE_SEED = 20261110
SAMPLE_MIN, SAMPLE_RATE = 5, 0.20
CANDIDATE_CAP = 10  # 방식별 후보가 이보다 많으면 같은 개수만 무작위로 확인(방식 간 확인 비율을 맞추기 위해, 사용자 승인 2026-11-10)
PAIRS = (("dense", "hybrid"), ("hybrid", "hybrid_rerank"), ("dense", "hybrid_rerank"))
CASE_METRICS = ("status_match", "faith", "j_facts", "j_conditions", "j_evidence", "j_decision")
CASES_PER_DIRECTION = 3


def has_zero(judge_row: dict | None) -> bool:
    """Judge(다수결) 항목 중 점수 0이 하나라도 있으면 True."""
    items = (judge_row or {}).get("items") or {}
    return any((items.get(k) or {}).get("status") == "scored" and (items.get(k) or {}).get("score") == 0 for k in JUDGE_ITEMS)


def classification_cases(runs: dict, refs: dict, judge: dict, set_of: dict) -> tuple[list[dict], dict]:
    """분류 대상(Judge에 0점이 있는 답변)과 제외 건수. 서비스 error 답변은 모델 답변이 없어 시스템 오류로 따로 센다."""
    cases, skipped = [], {"service_error": 0, "no_zero": 0}
    for (qid, method), rec in sorted(runs.items()):
        status = (rec.get("response") or {}).get("status")
        if status == "error":
            skipped["service_error"] += 1
            continue
        jr = judge.get(rec["request_id"])
        if not has_zero(jr):
            skipped["no_zero"] += 1
            continue
        ref = refs[qid]
        chunks = (rec.get("final_context") or {}).get("chunks", [])
        resp = rec.get("response") or {}
        cases.append({
            "request_id": rec["request_id"], "question_id": qid, "rag_method": method, "evaluation_set_type": set_of.get(qid),
            "question": rec.get("question"), "expected_action": ref["expected_action"], "required_facts": ref.get("required_facts") or [],
            "forbidden_claims": ref.get("forbidden_claims") or [], "status": status, "answer": resp.get("answer"),
            "citations": format_citations(resp.get("citations")), "source_table": format_source_table(chunks), "final_context": format_context(chunks),
            "judge": {k: {kk: (jr["items"].get(k) or {}).get(kk) for kk in ("votes", "status", "score", "reason")} for k in JUDGE_ITEMS},
            "context_check": jr.get("context_check")})
    return cases, skipped


def validate_classification(rows: list[dict], expected_ids: Iterable[str]) -> list[str]:
    """분류 기록이 모든 대상 요청을 한 번씩 담고 형식이 맞는지 확인한다. 문제 목록(빈 목록이면 통과)."""
    problems, seen = [], {}
    expected = set(expected_ids)
    for r in rows:
        rid = r.get("request_id")
        if rid in seen:
            problems.append(f"분류가 두 번 기록됨: {rid}")
        seen[rid] = r
        crit = r.get("criteria") or {}
        if set(crit) != set(CRITERIA):
            problems.append(f"{rid}: criteria 키가 C1~C5와 다름")
        for c in CRITERIA:
            e = crit.get(c) or {}
            if not isinstance(e.get("possible"), bool) or not isinstance(e.get("evidence"), str):
                problems.append(f"{rid}: {c}에 possible(bool)과 evidence(str)가 필요함")
        if r.get("cause") not in CAUSES:
            problems.append(f"{rid}: cause가 허용값이 아님({r.get('cause')})")
        if not r.get("reason") or not r.get("classified_by"):
            problems.append(f"{rid}: reason과 classified_by가 필요함")
    for rid in sorted(expected - set(seen)):
        problems.append(f"분류 누락: {rid}")
    for rid in sorted(set(seen) - expected):
        problems.append(f"대상이 아닌 요청이 분류됨: {rid}")
    return problems


def is_candidate(row: dict) -> bool:
    """후보 여부는 모델·분류자의 `candidate` 값이 아니라 criteria에서 다시 계산한다."""
    return any((row.get("criteria") or {}).get(c, {}).get("possible") for c in CRITERIA)


def sample_size(n_not_candidate: int) -> int:
    return min(n_not_candidate, max(SAMPLE_MIN, math.ceil(SAMPLE_RATE * n_not_candidate)))


def select_review(classified: list[dict], seed: int = SAMPLE_SEED, cap: int = CANDIDATE_CAP) -> dict:
    """사람이 확인할 대상: 후보(방식별 `cap`건 이하면 전부, 넘으면 같은 개수만 무작위) + '중대 아님' 중 방식별 무작위 표본. 같은 입력과 seed면 항상 같은 결과."""
    candidates: list[str] = []
    skipped: dict[str, list[str]] = {}
    sample: dict[str, list[str]] = {}
    for m in METHODS:
        cand = sorted(r["request_id"] for r in classified if r["rag_method"] == m and is_candidate(r))
        picked = sorted(random.Random(f"{seed}:cand:{m}").sample(cand, cap)) if len(cand) > cap else cand
        candidates += picked
        skipped[m] = sorted(set(cand) - set(picked))
        pool = sorted(r["request_id"] for r in classified if r["rag_method"] == m and not is_candidate(r))
        k = sample_size(len(pool))
        sample[m] = sorted(random.Random(f"{seed}:{m}").sample(pool, k)) if k else []
    counts = {}
    for m in METHODS:
        total = sum(1 for r in classified if r["rag_method"] == m and is_candidate(r))
        counts[m] = {"candidates": total, "candidates_reviewed": total - len(skipped[m]), "candidates_cap_applied": bool(skipped[m]),
                     "not_candidates": sum(1 for r in classified if r["rag_method"] == m and not is_candidate(r)), "sampled": len(sample[m])}
    return {"seed": seed, "candidate_cap": cap, "candidates": sorted(candidates), "candidates_not_reviewed": skipped, "sample": sample, "counts": counts}


def representative_cases(table: dict, per_direction: int = CASES_PER_DIRECTION) -> list[dict]:
    """방식 쌍별 좋아진·나빠진 대표 질문. 규칙(고정): 값이 모두 있는 지표의 차이(뒤−앞) 합이 크고 작은 순, 같으면 질문 ID 순.
    Judge 사실·조건·근거·판단, 상태 일치, Faithfulness를 쓰고 검색 지표는 쓰지 않는다(검색이 같아도 답이 달라질 수 있음)."""
    out = []
    for a, b in PAIRS:
        diffs = []
        for (qid, m), ra in table.items():
            if m != a or (qid, b) not in table:
                continue
            rb = table[(qid, b)]
            ds = [rb[k] - ra[k] for k in CASE_METRICS if ra.get(k) is not None and rb.get(k) is not None]
            if ds and sum(ds) != 0:
                diffs.append((sum(ds), qid))
        better = sorted((d for d in diffs if d[0] > 0), key=lambda x: (-x[0], x[1]))[:per_direction]
        worse = sorted((d for d in diffs if d[0] < 0), key=lambda x: (x[0], x[1]))[:per_direction]
        for direction, lst in (("improved", better), ("worsened", worse)):
            for score, qid in lst:
                out.append({"pair": f"{a}->{b}", "direction": direction, "question_id": qid, "score_change": round(score, 3)})
    return out
