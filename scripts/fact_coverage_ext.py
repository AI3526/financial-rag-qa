"""확장 브랜치: 저장된 검색 결과에서 '필요한 사실이 최종 5개 청크에 들어 있는가'(사실 숫자 포함률)를 계산해 비교한다 (재검색·모델 호출 없음).

사용:
    uv run python scripts/fact_coverage_ext.py --base ext-int-r-base-d20s20f20r20 --runs ext-int-r-v2b-d20s20f20r20 ext-int-s5-v2b-d20s20f20r20 ...

(2026-11-11 수정: 숫자 사이 쉼표를 무시하고 한 자리 정수를 제외한다. 이전 판은 표의 `22000`과 사실의 `22,000`을 다른 숫자로 세어 포함률을 낮게 쟀다.)

배경: 이 코퍼스는 같은 사실을 한 문서의 여러 쪽에 되풀이한다. 그래서 평가 파일이 정답으로 지정한 특정 쪽을 못 찾아도 같은 사실이 든 다른 쪽을 찾았을 수 있고,
쪽 단위 지표(Recall 등)는 그런 경우를 '못 찾음'으로 센다. 이 지표는 쪽 번호와 무관하게 내용으로 센다.
- 문항의 `required_facts`마다 그 사실 문장에 든 숫자(단위 포함)가 **모두** 최종 5개 청크 본문의 합집합에 있으면 그 사실은 포함된 것이다(공백 제외).
  숫자가 없는 사실은 세지 않는다. 문항 점수 = 포함된 사실 수 / 숫자가 있는 사실 수. 숫자가 있는 사실이 없는 문항은 해당 없음(None)이다.
- 한계: 숫자가 들어 있다고 그 숫자가 맞는 맥락(회사·기간)에서 나온 것임을 보장하지 않는다(우연히 같은 숫자일 수 있음). 쪽 단위 지표와 함께 본다.
- 각 결과 폴더의 `manifest.json`에서 색인 이름을 읽어 해당 청킹의 `chunks.jsonl`을 찾는다. 정답 파일은 채점에만 쓴다.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.compare import paired_stats  # noqa: E402

CHUNK_DIRS = {"chunk-v1": "processed_intermediate", "chunk-v2a-tailmerge": "processed_intermediate_v2a", "chunk-v2b-max600": "processed_intermediate_v2b",
              "chunk-v2c-tableblock": "processed_intermediate_v2c", "chunk-v2e-overlap1": "processed_intermediate_v2e", "chunk-v3a-tablerows": "processed_intermediate_v3a",
              "chunk-v2f-punctjoin": "processed_intermediate_v2f"}
NUM = re.compile(r"\d[\d,\.]*\s*(?:억원|백만원|%|퍼센트)?")
TUNE = {"development", "ext_dev"}


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def canon(s: str) -> str:
    """공백과 숫자 사이 쉼표를 없앤다. 표의 숫자는 `22000`, 사실 문장은 `22,000`처럼 적혀 있어도 같은 숫자로 센다."""
    return re.sub(r"(?<=\d),(?=\d)", "", norm(s))


def fact_numbers(fact: str) -> list[str]:
    """사실 문장 안의 숫자(단위 포함). 한 자리 정수(`1`, `2` 등 분기 번호 같은 값)는 변별력이 없어 제외한다."""
    out = []
    for x in NUM.findall(fact):
        c = canon(x).rstrip(",.")  # 문장부호로 끝난 숫자(`22,000,`)의 뒤 부호는 숫자가 아니다
        if re.search(r"\d", c) and not re.fullmatch(r"\d", c):
            out.append(c)
    return out


def coverage(text: str, facts: list[str]) -> float | None:
    """숫자가 있는 사실 중 숫자가 모두 text에 든 비율(쉼표·공백 무시). 숫자가 있는 사실이 없으면 None."""
    text = canon(text)
    nums = [n for n in (fact_numbers(f) for f in facts) if n]
    if not nums:
        return None
    return sum(1 for n in nums if all(x in text for x in n)) / len(nums)


def load_refs() -> dict[str, dict]:
    refs = {}
    for f in ("development", "additional"):
        for q in json.loads((ROOT / "data" / "finance_intermediate" / "evaluation" / f"{f}.json").read_text(encoding="utf-8")):
            refs[q["id"]] = q
    for q in json.loads((ROOT / "data" / "finance_intermediate" / "evaluation_ext" / "ext_full.json").read_text(encoding="utf-8")):
        refs[q["id"]] = q
    hard = ROOT / "data" / "finance_intermediate" / "evaluation_ext" / "hard_full.json"
    if hard.exists():
        for q in json.loads(hard.read_text(encoding="utf-8")):
            refs[q["id"]] = q
    return refs


def score_run(name: str, refs: dict) -> dict:
    """결과 폴더 하나의 (질문, 방식) → 사실 숫자 포함률."""
    d = ROOT / "results" / name
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    version = manifest["index"].split("__")[1]
    chunks = {}
    for line in (ROOT / "data" / CHUNK_DIRS[version] / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            c = json.loads(line)
            chunks[c["chunk_id"]] = norm(c["chunk_text"])
    out = {}
    for line in (d / "retrieval.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        text = "".join(chunks[c["chunk_id"]] for c in r["retrieved"])
        out[(r["question_id"], r["rag_method"])] = (r["set"], coverage(text, refs[r["question_id"]]["required_facts"]), r["total_ms"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=["dense", "hybrid", "hybrid_rerank"])
    args = ap.parse_args()
    refs = load_refs()
    base = score_run(args.base, refs)
    f = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    for m in args.methods:
        print(f"\n[{m}] 사실 숫자 포함률 (tune, 기준선 {args.base} 대비 문항별 비교)")
        for name in [args.base] + args.runs:
            cur = score_run(name, refs) if name != args.base else base
            a = {q: v[1] for (q, mm), v in base.items() if mm == m and v[0] in TUNE}
            b = {q: v[1] for (q, mm), v in cur.items() if mm == m and v[0] in TUNE}
            vals = [x for x in b.values() if x is not None]
            mean = sum(vals) / len(vals) if vals else None
            ms = [v[2] for (q, mm), v in cur.items() if mm == m and v[0] in TUNE]
            if name == args.base:
                print(f"  {name:44} 평균 {f(mean)} (n={len(vals)}) | {sum(ms) / len(ms):.0f}ms | 기준선")
                continue
            s = paired_stats(a, b)
            ci = "-" if s["ci95"] is None else f"[{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}]"
            print(f"  {name:44} 평균 {f(mean)} (n={len(vals)}) | {sum(ms) / len(ms):.0f}ms | 차이 {s['mean_diff'] if s['mean_diff'] is None else format(s['mean_diff'], '+.3f')} "
                  f"CI{ci} 좋아짐/나빠짐/같음 {s['better']}/{s['worse']}/{s['same']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
