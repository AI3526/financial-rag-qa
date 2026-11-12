"""확장 브랜치: 최종 비교(B vs B+분해)에서 선택한 질문들의 분해 질의, 최종 문맥의 쪽, 정답 근거 쪽, 답변을 마크다운으로 보여 준다 (읽기 전용).

사용: uv run python scripts/export_final_cases_ext.py --ids IF-ADD10 IF-ADD18 --out results/_view/final_cases.md
- 정답 근거 쪽과 필수 사실은 평가 파일에서 읽으며 사후 분석용이다. 최종 비교가 끝난 뒤의 해석이며 후보·규칙을 바꾸는 데 쓰지 않는다(EXTENSION_INTERMEDIATE_FINAL_PLAN.md §5).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fact_coverage_ext import load_refs  # noqa: E402
from src.evaluation.official import load_judge, load_runs  # noqa: E402

CONFIG = {"B": "B (분해 없음)", "P": "B+분해"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan", default="ext-int-s7-plan-hold-v1")
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        print(f"이미 있습니다(덮어쓰지 않음): {out}. (중단)")
        return 1
    refs = load_refs()
    plan = {}
    for line in (ROOT / "results" / args.plan / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            plan[r["question_id"]] = r["queries"]
    runs = {(c, i): load_runs([f"ext-int-final-{c}-r{i}"]) for c in "BP" for i in (1, 2, 3)}
    judges = {(c, i): load_judge([f"ext-int-final-{c}-r{i}-judge"]) for c in "BP" for i in (1, 2, 3)}
    lines = ["# 최종 비교 질문별 보기 (사후 분석용)", "", "정답 근거 쪽과 필수 사실은 평가 파일의 값이다. ✓ = 최종 문맥에 그 쪽이 들어 있음.", ""]
    for qid in args.ids:
        ref = refs[qid]
        gold = []
        for g in ref["evidence_groups"]:
            for e in g:
                if (e["book_id"], e["page"]) not in gold:
                    gold.append((e["book_id"], e["page"]))
        lines += [f"## {qid} ({ref['expected_action']})", "", f"**질문**: {next(iter(runs[('B', 1)].values())) and [r for (q, m), r in runs[('B', 1)].items() if q == qid][0]['question']}", ""]
        lines += ["**필수 사실(정답 기준)**:"] + [f"- {f}" for f in ref["required_facts"]] + [""]
        lines += ["**정답 근거 쪽**: " + ", ".join(f"{b} p{p}" for b, p in gold), ""]
        lines += ["**분해 질의(B+분해)**:"] + ([f"{i}. {q}" for i, q in enumerate(plan.get(qid, []), 1)] or ["- (분해 질의 없음: 원래 질문만 사용)"]) + [""]
        for c in "BP":
            lines += [f"### {CONFIG[c]}", ""]
            for i in (1, 2, 3):
                rec = [r for (q, m), r in runs[(c, i)].items() if q == qid][0]
                j = judges[(c, i)].get(rec["request_id"]) or {}
                sc = {k: ((j.get("items") or {}).get(k) or {}).get("score") for k in ("facts", "conditions", "decision", "evidence")}
                pages = [(x["book_id"], x["page"]) for x in rec["final_context"]["chunks"]]
                has = ", ".join(f"{b} p{p}{'✓' if (b, p) in pages else '✗'}" for b, p in gold)
                lines += [f"- **실행 {i}**: 상태 {rec['response']['status']} | 사실/조건/판단/근거 {sc['facts']}/{sc['conditions']}/{sc['decision']}/{sc['evidence']} | 정답 쪽 {has}"]
            rec = [r for (q, m), r in runs[(c, 1)].items() if q == qid][0]
            pages = [f"{x['book_id']} p{x['page']}" for x in rec["final_context"]["chunks"]]
            lines += ["", f"- 실행 1 문맥의 쪽(12개): {', '.join(pages)}", f"- 실행 1 답변: {rec['response']['answer']}", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"저장: {out} ({len(args.ids)}개 질문)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
