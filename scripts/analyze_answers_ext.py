"""확장 브랜치 답변 단계: 답변·Judge·Faithfulness 결과를 모아 분석한다 (모델 호출 없음).

사용:
    uv run python scripts/analyze_answers_ext.py --source ext-int-ans-base-001 --judge ext-int-ans-base-001-judge --faith ext-int-ans-base-001-faithfulness

- 기본 그룹은 tune(development + ext_dev)이다. confirm(additional + ext_hold)은 `--groups tune confirm`처럼 명시해야 계산한다(최종 후보 확인용).
- 내용: 기대 동작별 응답 상태(혼동표), Judge 4항목 충족률(점수가 있는 문항만, 해당 없음·평가 불가는 개수), Faithfulness 평균,
  기대 동작이 answer인데 사실 항목이 0점인 문항의 원인 분리(필수 사실이 최종 문맥에 있었는가: 생성 문제 vs 검색 문제), 문제 문항 목록.
- 결과는 `results/{source}-analysis/analysis.json`에도 저장한다(이미 있으면 중단).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.official import load_faithfulness, load_judge, load_runs  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from decompose_queries_ext import load_questions  # noqa: E402

GROUPS = {"tune": ("development", "ext_dev"), "confirm": ("additional", "ext_hold"), "development": ("development",), "additional": ("additional",),
          "ext_dev": ("ext_dev",), "ext_hold": ("ext_hold",), "hard_dev": ("hard_dev",), "hard_hold": ("hard_hold",), "cross_dev": ("cross_dev",), "cross_hold": ("cross_hold",)}
ITEMS = ("facts", "conditions", "evidence", "decision")
LABEL = {"facts": "사실", "conditions": "조건", "evidence": "근거", "decision": "판단"}
EXPECTED_STATUS = {"answer": "answered", "clarify": "needs_clarification", "abstain": "abstained"}


def load_refs() -> dict:
    refs = {}
    for f in ("development", "additional"):
        for q in json.loads((ROOT / "data" / "finance_intermediate" / "evaluation" / f"{f}.json").read_text(encoding="utf-8")):
            refs[q["id"]] = q
    for q in json.loads((ROOT / "data" / "finance_intermediate" / "evaluation_ext" / "ext_full.json").read_text(encoding="utf-8")):
        refs[q["id"]] = q
    for name in ("hard_full.json", "cross_full.json"):
        extra = ROOT / "data" / "finance_intermediate" / "evaluation_ext" / name
        if extra.exists():
            for q in json.loads(extra.read_text(encoding="utf-8")):
                refs[q["id"]] = q
    return refs


def mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--faith", required=True)
    ap.add_argument("--groups", nargs="+", default=["tune"], choices=sorted(GROUPS))
    args = ap.parse_args()
    out = ROOT / "results" / f"{args.source}-analysis"
    if out.exists():
        print(f"이미 결과가 있습니다(덮어쓰지 않음): {out}. (중단)")
        return 1
    refs = load_refs()
    set_of = {q["question_id"]: q["set"] for q in load_questions()}
    runs = load_runs([args.source])
    judge, faith = load_judge([args.judge]), load_faithfulness([args.faith])
    method = next(iter({m for _, m in runs}))
    f3 = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    result = {"source": args.source, "method": method, "groups": {}}
    for g in args.groups:
        sets = set(GROUPS[g])
        rows = []
        for (qid, m), rec in runs.items():
            if set_of[qid] not in sets:
                continue
            j = judge.get(rec["request_id"]) or {}
            rows.append({"qid": qid, "set": set_of[qid], "expected": refs[qid]["expected_action"], "status": (rec.get("response") or {}).get("status"), "rec": rec,
                         "items": j.get("items") or {}, "ctx": (j.get("context_check") or {}), "faith": faith.get(rec["request_id"]) or {}})
        print(f"\n######## 그룹 {g} (질문 {len(rows)}개: {dict(Counter(r['set'] for r in rows))})")
        # 1) 기대 동작 × 응답 상태
        print("\n[1] 기대 동작별 응답 상태")
        conf = {}
        for exp in ("answer", "clarify", "abstain"):
            rs = [r for r in rows if r["expected"] == exp]
            c = Counter(r["status"] for r in rs)
            match = sum(1 for r in rs if r["status"] == EXPECTED_STATUS[exp])
            conf[exp] = {"questions": len(rs), "statuses": dict(c), "match": match}
            print(f"   기대 {exp:8} {len(rs):3}개 → {dict(c)} | 상태 일치 {match}/{len(rs)}")
        # 2) Judge 항목
        print("\n[2] Judge(3회 다수결) 충족률 — 점수가 있는 문항 기준(해당 없음/평가 불가 제외)")
        judge_rates = {}
        for label, subset in (("전체", rows),) + tuple((f"기대 {e}", [r for r in rows if r["expected"] == e]) for e in ("answer", "clarify", "abstain")):
            cells, rec = [], {}
            for k in ITEMS:
                vals = [r["items"].get(k, {}).get("score") for r in subset if r["items"].get(k, {}).get("status") == "scored"]
                na = sum(1 for r in subset if r["items"].get(k, {}).get("status") == "not_applicable")
                rec[k] = {"rate": mean(vals), "n": len(vals), "not_applicable": na}
                cells.append(f"{LABEL[k]} {f3(mean(vals))}(n={len(vals)}" + (f", 해당없음 {na}" if na else "") + ")")
            judge_rates[label] = rec
            print(f"   {label:10} {len(subset):3}개 | " + " | ".join(cells))
        # 3) Faithfulness
        fv = [r["faith"].get("score") for r in rows if r["faith"].get("status") == "scored"]
        fna = sum(1 for r in rows if r["faith"].get("status") == "not_applicable")
        print(f"\n[3] Faithfulness 평균 {f3(mean(fv))} (n={len(fv)}, 해당 없음 {fna})")
        # 4) 사실 0점 원인 분리
        print("\n[4] 기대 동작이 answer인데 Judge 사실이 0점인 문항: 필수 사실이 최종 문맥에 있었는가")
        zero = [r for r in rows if r["expected"] == "answer" and r["items"].get("facts", {}).get("status") == "scored" and r["items"]["facts"]["score"] == 0]
        cov = Counter(r["ctx"].get("coverage") for r in zero)
        ok_answers = [r for r in rows if r["expected"] == "answer"]
        print(f"   answer 문항 {len(ok_answers)}개 중 사실 0점 {len(zero)}개 → 문맥 포함 정도 {dict(cov)} (all=필수 사실이 모두 문맥에 있음 → 생성 문제, some/none=일부·전부 없음 → 검색 문제)")
        print(f"   그중 응답 상태가 abstained/needs_clarification: {sum(1 for r in zero if r['status'] != 'answered')}개")
        # 5) 문제 문항
        print("\n[5] 문제 문항(answer 기대, 사실 또는 판단 0점)")
        problems = []
        for r in sorted(rows, key=lambda x: x["qid"]):
            if r["expected"] != "answer":
                continue
            fz = r["items"].get("facts", {}).get("score") == 0
            dz = r["items"].get("decision", {}).get("score") == 0
            if fz or dz:
                reason = (r["items"].get("facts", {}).get("reason") or "")[:70].replace("\n", " ")
                problems.append({"qid": r["qid"], "status": r["status"], "facts": r["items"].get("facts", {}).get("score"), "ctx": r["ctx"].get("coverage"), "reason": reason})
                print(f"   {r['qid']:11} 상태 {r['status']:19} 사실 {r['items'].get('facts', {}).get('score')} 문맥 {str(r['ctx'].get('coverage')):5} | {reason}")
        # 운영
        from src.evaluation.compare import operational

        ops = operational([r["rec"] for r in rows])
        print(f"\n[6] 운영(이 그룹): 평균 응답 {ops['mean_total_ms']:.0f}ms(검색 {ops['mean_retrieval_ms']:.0f} + 생성 {ops['mean_generation_ms']:.0f}), 입력 {ops['input_tokens']}·출력 {ops['output_tokens']}토큰, 비용 ${ops['cost_usd']:.4f}, 오류 {ops['errors']}")
        result["groups"][g] = {"confusion": conf, "judge": judge_rates, "faithfulness": {"mean": mean(fv), "n": len(fv), "not_applicable": fna},
                               "zero_facts_context": {"answer_questions": len(ok_answers), "zero_facts": len(zero), "context_coverage": dict(cov)}, "problems": problems, "operational": ops}
    out.mkdir(parents=True)
    (out / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
