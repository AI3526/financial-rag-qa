"""확장 브랜치: 답변 실행 결과(runs.jsonl)에서 질문·답변·재검색 정보를 사람이 읽기 쉬운 마크다운으로 뽑는다 (읽기 전용, 모델·검색 호출 없음).

사용:
    uv run python scripts/export_answers_md_ext.py --runs ext-int-ans-p40k12-004 ext-int-ans-p40k12-retry-broad-001 --ids IF-DEV07 IF-DEV19 --out results/_view/IF-DEV07.md
    (--ids를 생략하면 첫 번째 실행의 모든 질문)

- 질문 한 개마다 실행별로: 최종 상태·답변·이유, 재검색을 했다면 두 번째 검색 질의와 더한 청크(책, 쪽), 최종 문맥에 들어간 쪽 목록.
- 첫 번째 답변 전체는 저장돼 있지 않아(두 번째 생성이 덮어씀) 이유 문장은 `retry.query`에 든 것으로만 보인다.
- 정답 기준(필수 사실)은 평가 파일에서 읽어 질문 아래에 함께 적는다(읽는 사람의 비교용이며 모델에는 전달하지 않는다).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fact_coverage_ext import load_refs  # noqa: E402


def load_runs(name: str) -> dict[str, dict]:
    path = ROOT / "results" / name / "runs.jsonl"
    return {r["question_id"]: r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="results/ 아래 실행 폴더 이름들")
    ap.add_argument("--ids", nargs="+")
    ap.add_argument("--out", required=True, help="저장할 마크다운 경로(저장소 루트 기준)")
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        print(f"이미 있습니다(덮어쓰지 않음): {out}. (중단)")
        return 1
    runs = {n: load_runs(n) for n in args.runs}
    refs = load_refs()
    ids = args.ids or list(runs[args.runs[0]].keys())
    lines = ["# 답변 보기", "", "실행: " + ", ".join(f"`{n}`" for n in args.runs), ""]
    for qid in ids:
        ref = refs.get(qid, {})
        q = next((runs[n][qid]["question"] for n in args.runs if qid in runs[n]), "(질문 없음)")
        lines += [f"## {qid}", "", f"**질문**: {q}", ""]
        if ref.get("required_facts"):
            lines += ["**정답 기준(필수 사실)**:"] + [f"- {f}" for f in ref["required_facts"]] + [""]
        for n in args.runs:
            r = runs[n].get(qid)
            if not r:
                lines += [f"### {n}", "(이 실행에 없음)", ""]
                continue
            resp, retry = r["response"], r.get("retry") or {}
            lines += [f"### {n}", f"- 최종 상태: **{resp['status']}**"]
            if retry:
                lines.append(f"- 재검색: {'실행' if retry.get('triggered') else '안 함'} (첫 상태 {retry.get('first_status')})")
            if retry.get("triggered"):
                lines.append(f"- 두 번째 검색 질의: {retry['query']}")
                lines.append("- 더한 청크: " + ", ".join(f"{c['book_id']} p{c['page']}" for c in retry["added_chunks"]))
            chunks = (r.get("final_context") or {}).get("chunks") or []
            lines.append("- 최종 문맥의 쪽: " + ", ".join(f"{c['book_id']} p{c['page']}" for c in chunks))
            lines += [f"- 답변: {resp['answer']}", f"- 이유: {resp['reason']}", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"저장: {out} ({len(ids)}개 질문)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
