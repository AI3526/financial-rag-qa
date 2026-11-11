"""확장 브랜치: 두 검색 평가 결과 폴더를 문항별로 짝지어 비교한다 (사전 고정 규칙, EXTENSION_INTERMEDIATE_LOG).

사용:
    uv run python scripts/compare_retrieval_ext.py --base ext-int-retrieval-base-002 --cand ext-int-s1-d40s40f60r60 [--cand ...] --group tune

- 지표별로 평균 차이, 좋아진·나빠진·같은 문항 수, 부트스트랩 95% 신뢰구간(참고, 문턱 아님), ±0.05 기준 "비슷함" 표시를 낸다.
- 기본 그룹은 tune(development + ext_dev)이다. confirm 그룹은 `--group confirm`을 명시해야 비교한다(최종 후보 확인용).
- 값이 없는 문항(근거 없음, N/A)은 짝에서 제외한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.compare import paired_stats  # noqa: E402

GROUPS = {"tune": ("development", "ext_dev"), "confirm": ("additional", "ext_hold"), "development": ("development",), "additional": ("additional",),
          "ext_dev": ("ext_dev",), "ext_hold": ("ext_hold",)}
METRICS = ("page_recall", "page_hit", "page_mrr", "page_ndcg", "recall", "hit", "mrr", "ndcg")
METHODS = ("dense", "hybrid", "hybrid_rerank")


def load(name: str) -> dict:
    rows = [json.loads(l) for l in (ROOT / "results" / name / "retrieval.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return {(r["question_id"], r["rag_method"]): r for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", action="append", required=True)
    ap.add_argument("--group", default="tune", choices=sorted(GROUPS))
    ap.add_argument("--methods", nargs="+", default=["hybrid", "hybrid_rerank"], choices=METHODS)
    args = ap.parse_args()
    base = load(args.base)
    sets = set(GROUPS[args.group])
    f = lambda x: "-" if x is None else f"{x:+.3f}"  # noqa: E731
    for cand in args.cand:
        c = load(cand)
        print(f"\n## {cand} − {args.base} ({args.group})")
        for m in args.methods:
            print(f"  [{m}]")
            for k in METRICS:
                a = {q: r[k] for (q, mm), r in base.items() if mm == m and r["set"] in sets}
                b = {q: r[k] for (q, mm), r in c.items() if mm == m and r["set"] in sets}
                s = paired_stats(a, b)
                ci = "-" if s["ci95"] is None else f"[{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}]"
                print(f"    {k:12} {s['mean_a']:.3f} → {s['mean_b']:.3f} 차이 {f(s['mean_diff'])} CI{ci} 좋아짐/나빠짐/같음 {s['better']}/{s['worse']}/{s['same']} (n={s['n']}) "
                      f"{'비슷함' if s['similar'] else '**차이**'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
