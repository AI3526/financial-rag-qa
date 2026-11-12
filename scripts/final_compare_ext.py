"""확장 브랜치: 최종 비교(B vs B+분해, EXTENSION_INTERMEDIATE_FINAL_PLAN.md)를 사전 고정한 규칙대로 판정한다 (읽기 전용).

각 후보의 3회 실행 평균을 계산하고 선택 규칙 4개를 순서대로 적용한다. 계획에 없는 기준은 판정에 쓰지 않고 참고로만 출력한다.
사용: uv run python scripts/final_compare_ext.py --out results/ext-int-final-decision.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from decompose_queries_ext import load_questions  # noqa: E402
from fact_coverage_ext import load_refs  # noqa: E402
from src.evaluation.official import load_faithfulness, load_judge, load_runs  # noqa: E402

RUNS = {"B": [f"ext-int-final-B-r{i}" for i in (1, 2, 3)], "P": [f"ext-int-final-P-r{i}" for i in (1, 2, 3)]}
ITEMS = ("facts", "conditions", "evidence", "decision")
ONE_Q = 1 / 53
EPS = 1e-9  # 1질문 경계(정확히 1/53 차이)가 부동소수점 오차로 '넘는다'로 판정되지 않게 하는 허용 오차
SET_OF = {q["question_id"]: q["set"] for q in load_questions()}
LIKE_19 = {"IF-ADD16", "IF-ADD17", "IF-ADD18", "IF-ADD23"}
CALC = {"IF-ADD21", "IF-ADD22", "IF-ADD24", "IF-ADD25", "IF-ADD26", "IF-ADD27"} | {f"IFH-H{i:02d}" for i in range(1, 11)} | {f"IFX2-H{i:02d}" for i in range(1, 5)}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def run_stats(name: str, refs: dict) -> dict:
    runs = load_runs([name])
    judge, faith = load_judge([f"{name}-judge"]), load_faithfulness([f"{name}-faithfulness"])
    rows = []
    for (qid, _method), rec in runs.items():
        j = judge.get(rec["request_id"]) or {}
        items = {k: ((j.get("items") or {}).get(k) or {}).get("score") for k in ITEMS}
        f = faith.get(rec["request_id"]) or {}
        rows.append({"qid": qid, "set": SET_OF[qid], "expected": refs[qid]["expected_action"], "status": rec["response"]["status"], "items": items,
                     "faith": f.get("score") if f.get("status") == "scored" else None, "ms": rec["performance"]["total_duration_ms"],
                     "cost": rec.get("cost") or 0.0, "in_tok": sum((c.get("input_tokens") or 0) for c in rec["llm_calls"])})
    return {"rows": rows}


def aggregate(stats: list[dict], pred=lambda r: True) -> dict:
    per_run = []
    for s in stats:
        rows = [r for r in s["rows"] if pred(r)]
        per_run.append({**{k: mean([r["items"][k] for r in rows]) for k in ITEMS}, "faith": mean([r["faith"] for r in rows]),
                        "abstain_match": sum(1 for r in rows if r["expected"] == "abstain" and r["status"] == "abstained"),
                        "n_abstain": sum(1 for r in rows if r["expected"] == "abstain"), "ms": mean([r["ms"] for r in rows]),
                        "cost": sum(r["cost"] for r in rows), "in_tok": sum(r["in_tok"] for r in rows), "n": len(rows)})
    keys = ("facts", "conditions", "evidence", "decision", "faith", "abstain_match", "ms", "cost", "in_tok")
    out = {k: mean([p[k] for p in per_run]) for k in keys}
    out["n"], out["n_abstain"] = per_run[0]["n"], per_run[0]["n_abstain"]
    out["runs"] = per_run
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        print(f"이미 있습니다(덮어쓰지 않음): {out}. (중단)")
        return 1
    refs = load_refs()
    st = {c: [run_stats(n, refs) for n in names] for c, names in RUNS.items()}
    agg = {c: aggregate(s) for c, s in st.items()}
    f = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"  # noqa: E731
    print(f"{'':10}{'사실':>8}{'조건':>8}{'근거':>8}{'판단':>8}{'Faith':>8}{'abstain일치':>12}{'응답ms':>9}{'비용$':>9}")
    for c in ("B", "P"):
        a = agg[c]
        print(f"{'B' if c == 'B' else 'B+분해':10}{f(a['facts']):>8}{f(a['conditions']):>8}{f(a['evidence']):>8}{f(a['decision']):>8}{f(a['faith']):>8}{f(a['abstain_match'], 2) + '/' + str(a['n_abstain']):>12}{f(a['ms'], 0):>9}{f(a['cost'], 4):>9}")
    b, p = agg["B"], agg["P"]
    rules = {
        "1 사실: B+분해가 B보다 1질문 넘게 높다": (p["facts"] - b["facts"]) > ONE_Q + EPS,
        "2a 조건: B+분해가 B보다 1질문 넘게 낮지 않다": (b["conditions"] - p["conditions"]) <= ONE_Q + EPS,
        "2b 판단: B+분해가 B보다 1질문 넘게 낮지 않다": (b["decision"] - p["decision"]) <= ONE_Q + EPS,
        "3 Faithfulness: B+분해가 B보다 0.03 넘게 낮지 않다": (b["faith"] - p["faith"]) <= 0.03 + EPS,
        "4 abstain 일치: B+분해가 B보다 낮지 않다": p["abstain_match"] >= b["abstain_match"],
    }
    print("\n사전 고정 선택 규칙(모두 참이어야 B+분해):")
    for k, v in rules.items():
        print(f"  [{'O' if v else 'X'}] {k}")
    verdict = "B+분해" if all(rules.values()) else "B"
    print(f"\n==> 선택: {verdict}")
    print(f"차이(B+분해 − B): 사실 {p['facts'] - b['facts']:+.3f}({(p['facts'] - b['facts']) / ONE_Q:+.1f}질문) 조건 {p['conditions'] - b['conditions']:+.3f} 판단 {p['decision'] - b['decision']:+.3f} Faith {p['faith'] - b['faith']:+.3f}")
    print("\n[참고] 실행별 사실/조건/판단/Faith")
    for c in ("B", "P"):
        for i, r in enumerate(agg[c]["runs"], 1):
            print(f"  {c}-r{i}: {f(r['facts'])} / {f(r['conditions'])} / {f(r['decision'])} / {f(r['faith'])}")
    print("\n[참고] 세트별·부분집합별 (사실/조건/판단/Faith, 질문 수)")
    groups = {"additional": lambda r: r["set"] == "additional", "ext_hold": lambda r: r["set"] == "ext_hold", "hard_hold": lambda r: r["set"] == "hard_hold",
              "cross_hold": lambda r: r["set"] == "cross_hold", "DEV19와 매우 비슷(ADD16·17·18·23)": lambda r: r["qid"] in LIKE_19,
              "계산·집계형": lambda r: r["qid"] in CALC}
    detail = {}
    for g, pred in groups.items():
        ga, gb = aggregate(st["P"], pred), aggregate(st["B"], pred)
        detail[g] = {"B": {k: gb[k] for k in ("facts", "conditions", "decision", "faith", "n")}, "P": {k: ga[k] for k in ("facts", "conditions", "decision", "faith", "n")}}
        print(f"  {g:28} B  {f(gb['facts'])}/{f(gb['conditions'])}/{f(gb['decision'])}/{f(gb['faith'])} (n={gb['n']})   B+분해 {f(ga['facts'])}/{f(ga['conditions'])}/{f(ga['decision'])}/{f(ga['faith'])}")
    # 질문별 차이(3회 평균 사실 점수)
    print("\n[참고] 질문별 사실 점수 3회 평균이 B와 B+분해에서 다른 문항")
    qs = sorted({r["qid"] for r in st["B"][0]["rows"]})
    diffs = []
    for q in qs:
        vb = mean([r["items"]["facts"] for s in st["B"] for r in s["rows"] if r["qid"] == q])
        vp = mean([r["items"]["facts"] for s in st["P"] for r in s["rows"] if r["qid"] == q])
        if vb is not None and vp is not None and abs(vb - vp) > 1e-9:
            diffs.append({"qid": q, "B": vb, "P": vp})
            print(f"  {q:11} B {vb:.2f}  B+분해 {vp:.2f}  ({vp - vb:+.2f})")
    out.write_text(json.dumps({"verdict": verdict, "rules": rules, "aggregate": {c: {k: v for k, v in a.items()} for c, a in agg.items()}, "detail": detail, "question_diffs": diffs},
                              ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
