"""STEP 12: 공식 비교 결과표 (12_ACCEPTANCE_CRITERIA §3·§4·§8, DECISIONS D-045).

사용:
    uv run python scripts/compare_official.py --source official-001 --retrieval official-001-retrieval \\
        --faithfulness official-001-faithfulness --judge official-001-judge --out official-001-compare

- 입력은 서비스가 남긴 `runs.jsonl`과 세 평가의 결과 폴더(여러 폴더를 줄 수 있음)다. 정답 파일은 질문별 기대 동작을 읽는 데만 쓴다.
- 지표를 **하나의 점수로 합산하지 않는다**(검색 지표, Faithfulness, Judge 네 항목을 각각 보고).
- 값이 없는 문항(해당 없음, 평가 불가)은 평균에서 빼고 개수를 적는다. 0점은 평균에 포함한다.
- **additional을 주결과**, development는 참고 결과로 분리해 보고한다. 기대 동작별로도 따로 본다.
- 방식 쌍별로 평균 차이와 **좋아진·나빠진·같은 문항 수**, 부트스트랩 95% 신뢰구간(**참고용, 선택의 문턱 아님**), "비슷함"(±0.05) 표시를 낸다.
- 평가 폴더의 manifest가 동결된 평가 설정(프롬프트 해시, 3회, 출력 한도, ragas 버전)과 같은지 확인해 결과에 적는다. 다르면 **경고를 표시**한다.
- 이 스크립트는 **최종 방식을 정하지 않는다.** 선택은 사전 확정된 규칙(§4.3)을 중대 오류 분류와 함께 적용해 사람이 확인한다.
- 결과 폴더가 이미 있으면 덮어쓰지 않고 중단한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.compare import SIMILAR, group_stats, operational, paired_stats  # noqa: E402
from src.evaluation.judge import load_references  # noqa: E402
from src.evaluation.official import (FREEZE_PATH, METHODS, METRICS, build_table, load_faithfulness, load_judge, load_retrieval,  # noqa: E402
                                     load_runs, question_set_of, verify_evaluation_manifest)

PAIRS = (("dense", "hybrid"), ("hybrid", "hybrid_rerank"), ("dense", "hybrid_rerank"))
GROUPS = (("additional", lambda r: r["evaluation_set_type"] == "additional"), ("development", lambda r: r["evaluation_set_type"] == "development"),
          ("all", lambda r: True))
ACTIONS = ("answer", "clarify", "abstain")
LABEL = dict(METRICS)


def fmt(v, digits=3):
    return "-" if v is None else f"{v:.{digits}f}"


def method_stats(table: dict, rows_filter) -> dict:
    out = {}
    for m in METHODS:
        rows = {k[0]: r for k, r in table.items() if k[1] == m and rows_filter(r)}
        out[m] = {key: group_stats({q: r[key] for q, r in rows.items()}) for key, _ in METRICS}
        out[m]["_questions"] = len(rows)
    return out


def pair_stats(table: dict, rows_filter) -> dict:
    out = {}
    for a, b in PAIRS:
        ra = {k[0]: r for k, r in table.items() if k[1] == a and rows_filter(r)}
        rb = {k[0]: r for k, r in table.items() if k[1] == b and rows_filter(r)}
        out[f"{a}->{b}"] = {key: paired_stats({q: r[key] for q, r in ra.items()}, {q: r[key] for q, r in rb.items()}) for key, _ in METRICS}
    return out


def render(res: dict) -> str:
    L = [f"# 공식 비교 결과표 ({res['experiment']})", "",
         "> 이 표는 **최종 방식을 정하지 않습니다.** 선택은 `12_ACCEPTANCE_CRITERIA` §4.3(품질 비교 → 중대 오류 → 품질이 비슷하면 운영 지표·복잡도)을 중대 오류 분류와 함께 적용해 사람이 확인합니다. "
         "점수를 하나로 합산하지 않으며, 값이 없는 문항(해당 없음·평가 불가)은 평균에서 빼고 개수를 적었습니다. 평균 옆 `n`은 값이 있는 문항 수, `없음`은 값이 없는 문항 수입니다.",
         "> 표본이 작아(세트당 30문항, 한 문제가 0.033) **±0.05 이내의 차이는 우연과 구분되지 않을 수 있고**, 신뢰구간은 참고용입니다. 제공 질문과 정답은 공개되어 있어 결과를 비공개 테스트 성능으로 표현하지 않습니다. "
         "평가 모델이 생성 모델과 같아 자기 평가 편향 가능성이 있습니다.", ""]
    L += ["## 0. 동결·평가 설정 확인", ""]
    for line in res["checks"]:
        L.append(f"- {line}")
    L += ["", f"- 질문 {res['questions']}개 × 방식 {len(METHODS)}개 = 기대 {res['expected']}건, 기록 {res['found']}건"]
    for gname, _ in GROUPS:
        L += ["", f"## 1. 방식별 품질 — {gname}" + (" (주결과)" if gname == "additional" else " (참고)" if gname == "development" else " (전체, 참고)"), "",
              "| 지표 | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
        for key, label in METRICS:
            cells = []
            for m in METHODS:
                s = res["by_group"][gname][m][key]
                cells.append(f"{fmt(s['mean'])} (n={s['n']}, 없음 {s['missing']})")
            L.append(f"| {label} | " + " | ".join(cells) + " |")
    L += ["", "## 2. 기대 동작별 (전체 60문항)", ""]
    for act in ACTIONS:
        L += [f"### 기대 동작 `{act}`", "", "| 지표 | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
        for key, label in METRICS:
            cells = []
            for m in METHODS:
                s = res["by_action"][act][m][key]
                cells.append(f"{fmt(s['mean'])} (n={s['n']}, 없음 {s['missing']})")
            L.append(f"| {label} | " + " | ".join(cells) + " |")
        L.append("")
    for gname, _ in GROUPS:
        L += [f"## 3. 방식 쌍별 차이 — {gname}" + (" (주결과)" if gname == "additional" else ""), "",
              f"`차이` = 뒤 방식 − 앞 방식. `비슷함`은 평균 차이가 ±{SIMILAR} 이내(설명용 기준). `좋아짐/나빠짐/같음`은 뒤 방식 기준 문항 수. 신뢰구간은 참고용입니다.", ""]
        for pair, stats in res["pairs"][gname].items():
            L += [f"### {pair.replace('->', ' → ')}", "", "| 지표 | 평균(앞) | 평균(뒤) | 차이 | 95% 신뢰구간(참고) | 좋아짐/나빠짐/같음 | 짝 지은 문항 | 비슷함 |", "|---|---|---|---|---|---|---|---|"]
            for key, label in METRICS:
                s = stats[key]
                ci = "-" if s["ci95"] is None else f"[{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]"
                sim = "-" if s["similar"] is None else ("예" if s["similar"] else "**아니오**")
                L.append(f"| {label} | {fmt(s['mean_a'])} | {fmt(s['mean_b'])} | {'-' if s['mean_diff'] is None else f'{s['mean_diff']:+.3f}'} | {ci} | "
                         f"{s['better']}/{s['worse']}/{s['same']} | {s['n']} | {sim} |")
            L.append("")
    L += ["## 4. 운영 지표 (일반 평가 요청만, 평가 호출·보안·모의 오류 제외)", "",
          "| 지표 | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    rows = [("요청 수", "requests", lambda v: str(v)), ("정상 처리", "ok", str), ("시스템 오류", "errors", str), ("오류율", "error_rate", fmt),
            ("평균 응답시간(ms, 정상 건)", "mean_total_ms", lambda v: "-" if v is None else f"{v:.0f}"),
            ("평균 검색 시간(ms)", "mean_retrieval_ms", lambda v: "-" if v is None else f"{v:.0f}"),
            ("평균 생성 시간(ms)", "mean_generation_ms", lambda v: "-" if v is None else f"{v:.0f}"),
            ("입력 토큰 합계(확인한 요청만)", "input_tokens", str), ("출력 토큰 합계(확인한 요청만)", "output_tokens", str),
            ("사용량 확인 요청 / 확인 불가", None, None), ("비용 합계(USD, 확인 가능한 요청만)", "cost_usd", lambda v: f"{v:.4f}"),
            ("비용 확인 요청 / 확인 불가", None, None), ("재시도 수", "retries", str)]
    for label, key, f in rows:
        if key is None:
            usage = label.startswith("사용량")
            cells = [(f"{res['ops'][m]['usage_known_requests']} / {res['ops'][m]['usage_unknown_requests']}" if usage
                      else f"{res['ops'][m]['cost_known_requests']} / {res['ops'][m]['cost_unknown_requests']}") for m in METHODS]
        else:
            cells = [f(res["ops"][m][key]) for m in METHODS]
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L += ["", "운영 지표는 요청을 한 번에 하나씩 보낸 실행에서 얻었습니다(동시성 1). 사용량·비용을 확인하지 못한 요청은 0이 아니라 확인 불가로 세었습니다.", ""]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", nargs="+", required=True, help="서비스 실행 폴더(runs.jsonl)")
    ap.add_argument("--retrieval", nargs="+", required=True)
    ap.add_argument("--faithfulness", nargs="+", required=True)
    ap.add_argument("--judge", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-freeze-check", action="store_true", help="평가 manifest를 동결과 대조하지 않음(개발용 소량 확인)")
    args = ap.parse_args()
    out = ROOT / "results" / args.out
    if out.exists():
        print(f"이미 결과가 있습니다(덮어쓰지 않음): {out}. (중단)")
        return 1

    runs = load_runs(args.source)
    refs = load_references()
    set_of = question_set_of()
    table = build_table(runs, refs, load_retrieval(args.retrieval), load_faithfulness(args.faithfulness), load_judge(args.judge), set_of)

    checks: list[str] = []
    if args.no_freeze_check:
        checks.append("동결 대조를 하지 않았습니다(`--no-freeze-check`, 개발용).")
    else:
        frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        for kind, folders in (("judge", args.judge), ("faithfulness", args.faithfulness)):
            for f in folders:
                mf = json.loads((ROOT / "results" / f / "manifest.json").read_text(encoding="utf-8"))
                probs = verify_evaluation_manifest(kind, mf, frozen)
                checks.append(f"{kind} `{f}`: " + ("동결된 설정과 같음(프롬프트 해시, 회차, 출력 한도, 모델)" if not probs else "**경고** " + "; ".join(probs)))
    qids = {q for q, _ in table}
    questions = len(qids)
    methods_found = {m for _, m in table}
    expected = questions * len(methods_found)
    res = {"experiment": args.source[0] if len(args.source) == 1 else "+".join(args.source), "checks": checks, "questions": questions,
           "expected": expected, "found": len(table),
           "by_group": {g: method_stats(table, f) for g, f in GROUPS},
           "by_action": {a: method_stats(table, lambda r, a=a: r["expected_action"] == a) for a in ACTIONS},
           "pairs": {g: pair_stats(table, f) for g, f in GROUPS},
           "ops": {m: operational([r for (q, mm), r in runs.items() if mm == m]) for m in METHODS}}
    out.mkdir(parents=True)
    (out / "comparison.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out / "per_question.jsonl").open("w", encoding="utf-8") as fh:
        for k in sorted(table):
            fh.write(json.dumps(table[k], ensure_ascii=False) + "\n")
    (out / "comparison.md").write_text(render(res), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"out": args.out, "sources": args.source, "retrieval": args.retrieval, "faithfulness": args.faithfulness,
                                                   "judge": args.judge, "similar_threshold": SIMILAR, "bootstrap": {"n_boot": 10000, "seed": 20261110},
                                                   "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    print(f"질문 {questions}개, 기록 {len(table)}건 → {out}")
    for line in checks:
        print(" ", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
