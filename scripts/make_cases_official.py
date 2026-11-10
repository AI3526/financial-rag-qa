"""STEP 12: 중대 오류 분류 입력, 사람 확인 작업지, 대표 사례 작업지 (12_ACCEPTANCE_CRITERIA §5, DECISIONS D-045).

두 단계로 쓴다.
    1) 분류 입력 만들기 (Judge 0점 답변 전부):
       uv run python scripts/make_cases_official.py export --source official-001 --judge official-001-judge --out official-001-critical
       → classify_input.jsonl / classify_input.md / classify_meta.json
       이 세션의 Claude가 입력을 읽고 `critical_classification.jsonl`을 직접 작성한다(모델 호출 없음, `classified_by`에 주체를 적는다).
    2) 사람 확인 작업지 만들기 (후보(방식별 10건 초과 시 같은 개수만 무작위) + 방식별 무작위 표본, 대표 사례):
       uv run python scripts/make_cases_official.py worksheet --source official-001 --compare official-001-compare --out official-001-critical
       → review_selection.json / worksheet_critical.md / worksheet_cases.md / critical_summary.json

- 이미 있는 파일은 덮어쓰지 않는다(사람이 채운 작업지를 지키기 위해).
- 표본은 `random.Random("{seed}:{방식}")`로 뽑아 같은 입력이면 항상 같다. seed는 20261110.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.critical import (CRITERIA, CRITERIA_LABEL, classification_cases, representative_cases,  # noqa: E402
                                     select_review, validate_classification)
from src.evaluation.judge import load_references  # noqa: E402
from src.evaluation.official import METHODS, load_judge, load_runs, question_set_of  # noqa: E402

RESULTS = ROOT / "results"


def _write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"이미 있는 파일은 덮어쓰지 않습니다: {path}")
    path.write_text(text, encoding="utf-8")


def _jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def case_md(c: dict, with_context: bool = True) -> str:
    jd = "\n".join(f"  - {k}: {v['score'] if v['status'] == 'scored' else v['status']} (votes {v['votes']}) — {v['reason']}" for k, v in c["judge"].items())
    L = [f"### {c['question_id']} · {c['rag_method']} · 요청 `{c['request_id']}`", "",
         f"- 질문: {c['question']}", f"- 기대 동작: {c['expected_action']} / 서비스 상태: {c['status']}",
         f"- 필수 사실: {' | '.join(f'({i}) {f}' for i, f in enumerate(c['required_facts'], 1)) or '(없음)'}",
         f"- 금지 주장: {' | '.join(c['forbidden_claims']) or '(없음)'}", "", f"**답변**: {c['answer']}", "", f"**인용**: {c['citations']}", "",
         f"**Judge(3회 다수결)**:\n{jd}", ""]
    if c.get("context_check"):
        L += [f"**문맥 진단**: {json.dumps(c['context_check'].get('facts'), ensure_ascii=False)}", ""]
    L += [f"**출처 표**:\n{c['source_table']}", ""]
    if with_context:
        L += ["<details><summary>최종 문맥 전문</summary>", "", c["final_context"], "", "</details>", ""]
    return "\n".join(L)


def cmd_export(args) -> int:
    out = RESULTS / args.out
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.source)
    cases, skipped = classification_cases(runs, load_references(), load_judge(args.judge), question_set_of())
    _write_new(out / "classify_input.jsonl", _jsonl(cases))
    _write_new(out / "classify_input.md", "# 중대 오류 분류 입력 (Judge에 0점이 있는 답변)\n\n" + "\n---\n\n".join(case_md(c) for c in cases) + "\n")
    meta = {"requests": len(runs), "to_classify": len(cases), "skipped": skipped,
            "by_method": {m: sum(1 for c in cases if c["rag_method"] == m) for m in METHODS}}
    _write_new(out / "classify_meta.json", json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(f"요청 {len(runs)}건 중 분류 대상 {len(cases)}건 {meta['by_method']}, 제외 {skipped} → {out}")
    return 0


def cmd_worksheet(args) -> int:
    out = RESULTS / args.out
    cases = {json.loads(l)["request_id"]: json.loads(l) for l in (out / "classify_input.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    rows = [json.loads(l) for l in (out / "critical_classification.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    problems = validate_classification(rows, cases)
    if problems:
        print(f"분류 기록 검증 실패 ({len(problems)}건):")
        for p in problems[:30]:
            print("  -", p)
        return 1
    classified = [{**r, "rag_method": cases[r["request_id"]]["rag_method"], "question_id": cases[r["request_id"]]["question_id"]} for r in rows]
    sel = select_review(classified)
    by_id = {r["request_id"]: r for r in classified}
    review_ids = sorted(set(sel["candidates"]) | {i for v in sel["sample"].values() for i in v}, key=lambda i: (by_id[i]["rag_method"], by_id[i]["question_id"]))

    L = ["# 중대 오류 사람 확인 작업지", "",
         "각 사례의 **답변이 아래 다섯 기준 중 하나라도 해당하는지** 원문 PDF와 대조해 판단합니다. AI(이 세션의 Claude)의 분류와 근거를 먼저 읽을 수 있지만 **따르지 않아도 됩니다.**",
         "", *(f"- {c} {CRITERIA_LABEL[c]}" for c in CRITERIA), "",
         "- 분류 구분: `후보` = AI가 기준 중 하나라도 해당 가능성을 봄 / `표본` = AI가 중대 아님으로 분류했고 무작위로 뽑힘.",
         f"- 방식별 후보가 {sel['candidate_cap']}건을 넘으면 같은 개수만 무작위로 뽑았습니다(방식 간 확인 비율을 맞추기 위해). 적용된 방식: {', '.join(m for m in METHODS if sel['counts'][m]['candidates_cap_applied']) or '없음'}. 이 경우 결과는 표본 확인이며 전수가 아닙니다.",
         "- 표본에서 사람이 중대로 판정한 건이 하나라도 나오면 그 방식의 나머지 '중대 아님' 분류는 전부 사람이 확인합니다.", "",
         "표시 방법: `사람 판정:` 뒤에 `중대` / `중대 아님` / `판단 어려움`, `유형:`에 해당 기준 번호, `메모:`에 근거를 적습니다.", ""]
    for rid in review_ids:
        r, c = by_id[rid], cases[rid]
        kind = "후보" if rid in sel["candidates"] else "표본"
        crit = "\n".join(f"  - {k} {CRITERIA_LABEL[k]}: {'해당 가능' if r['criteria'][k]['possible'] else '해당 없음'} — {r['criteria'][k]['evidence']}" for k in CRITERIA)
        L += [f"## [{kind}] {c['question_id']} · {c['rag_method']}", "", case_md(c).split("\n", 2)[2],
              f"**AI 분류**(분류 주체: {r['classified_by']}): 원인 `{r['cause']}`\n{crit}\n- 요약: {r['reason']}", "",
              "- 사람 판정: \n- 유형: \n- 메모: \n", "---", ""]
    _write_new(out / "worksheet_critical.md", "\n".join(L))
    _write_new(out / "review_selection.json", json.dumps(sel, ensure_ascii=False, indent=2) + "\n")
    summary = {"classified": len(rows), "classified_by": sorted({r["classified_by"] for r in rows}), "counts": sel["counts"],
               "review_requests": len(review_ids), "note": "사람이 확인한 건수는 작업지 작성 후 집계한다. counts의 candidates는 AI만 분류한 후보 전체 수이고 candidates_reviewed는 사람이 확인하는 수다."}
    _write_new(out / "critical_summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"분류 {len(rows)}건, 사람 확인 {len(review_ids)}건(후보 {len(sel['candidates'])} + 표본) → {out / 'worksheet_critical.md'}")

    if args.compare:
        table_rows = [json.loads(l) for l in (RESULTS / args.compare / "per_question.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        table = {(r["question_id"], r["rag_method"]): r for r in table_rows}
        runs = load_runs(args.source)
        reps = representative_cases(table)
        M = ["# 대표 사례 작업지 (좋아진·나빠진 사례)", "",
             "선정 규칙(고정): 방식 쌍별로 상태 일치·Faithfulness·Judge 4항목의 차이(뒤−앞) 합이 큰 순(좋아짐)·작은 순(나빠짐) 3건씩, 같으면 질문 ID 순. 검색 지표는 쓰지 않습니다. "
             "각 사례에서 **자동 채점이 맞는지**, **답변 차이의 원인**(검색, 문맥, 생성)을 원문과 대조해 적습니다. 자동 채점 오류를 발견하면 원래 점수와 수정 이유를 남깁니다.", ""]
        for rep in reps:
            a, b = rep["pair"].split("->")
            M += [f"## {rep['pair'].replace('->', ' → ')} · {'좋아짐' if rep['direction'] == 'improved' else '나빠짐'} · {rep['question_id']} (변화 {rep['score_change']:+})", ""]
            for m in (a, b):
                row, rec = table[(rep["question_id"], m)], runs[(rep["question_id"], m)]
                vals = {k: row[k] for k in ("status_match", "hit", "recall", "faith", "j_facts", "j_conditions", "j_evidence", "j_decision")}
                M += [f"- **{m}** 상태 {row['status']} / 점수 {vals}", f"  - 답변: {(rec.get('response') or {}).get('answer')}"]
            M += ["", "- 자동 채점이 맞는가: ", "- 차이의 원인: ", "- 메모: ", "", "---", ""]
        _write_new(out / "worksheet_cases.md", "\n".join(M))
        _write_new(out / "cases_selection.json", json.dumps(reps, ensure_ascii=False, indent=2) + "\n")
        print(f"대표 사례 {len(reps)}건 → {out / 'worksheet_cases.md'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--source", nargs="+", required=True)
    e.add_argument("--judge", nargs="+", required=True)
    e.add_argument("--out", required=True)
    w = sub.add_parser("worksheet")
    w.add_argument("--source", nargs="+", required=True)
    w.add_argument("--compare", help="compare_official.py 결과 폴더(대표 사례용)")
    w.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        return cmd_export(args) if args.cmd == "export" else cmd_worksheet(args)
    except FileExistsError as e:
        print(e, "(중단)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
