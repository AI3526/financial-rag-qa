"""확장 브랜치 5a: 질문을 검색 질의로 분해해 저장한다 (모델 호출, Langfuse 미사용).

사용: uv run python scripts/decompose_queries_ext.py --out ext-int-s5-queries-v1

- 대상: Intermediate 제공 질문(development 30 + additional 30)과 자체 질문(`evaluation_ext`, 동결) 전체. **질문 번호와 질문 문자열만** 모델에 보낸다.
- 프롬프트: `prompts/query_decompose_v1.json`(사용자 승인 문안). 모델 호출은 질문당 1회이며 결과를 저장해 평가에서 재사용한다(같은 질문을 다시 호출하지 않음).
- 결과: `results/{out}/queries.jsonl`(질문별 추가 질의·원문 출력·사용량·시간·오류), `summary.json`, `manifest.json`(프롬프트 해시, 모델, 설정).
- 결과 폴더가 이미 있으면 덮어쓰지 않고 중단한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_env  # noqa: E402

load_env()

from src.retrieval import query_decompose as qd  # noqa: E402

EVAL = ROOT / "data" / "finance_intermediate" / "evaluation"
EXT = ROOT / "data" / "finance_intermediate" / "evaluation_ext"
SETS = (("development", "development_questions.json"), ("additional", "additional_questions.json"))


def load_questions() -> list[dict]:
    """질문 번호·질문·세트만 읽는다(정답 파일은 읽지 않음)."""
    out = []
    for set_name, f in SETS:
        out += [{"question_id": q["id"], "question": q["question"], "set": set_name} for q in json.loads((EVAL / f).read_text(encoding="utf-8"))]
    full = {x["id"]: x["split"] for x in json.loads((EXT / "ext_full.json").read_text(encoding="utf-8"))}
    out += [{"question_id": q["id"], "question": q["question"], "set": "ext_" + full[q["id"]]} for q in json.loads((EXT / "ext_questions.json").read_text(encoding="utf-8"))]
    if (EXT / "hard_questions.json").exists():  # 어려운 질문 세트(IFH, hard_dev·hard_hold)
        hard = {x["id"]: x["split"] for x in json.loads((EXT / "hard_full.json").read_text(encoding="utf-8"))}
        out += [{"question_id": q["id"], "question": q["question"], "set": "hard_" + hard[q["id"]]} for q in json.loads((EXT / "hard_questions.json").read_text(encoding="utf-8"))]
    if (EXT / "cross_questions.json").exists():  # 문서 건너뛰기 질문 세트(IFX2, cross_dev·cross_hold)
        cross = {x["id"]: x["split"] for x in json.loads((EXT / "cross_full.json").read_text(encoding="utf-8"))}
        out += [{"question_id": q["id"], "question": q["question"], "set": "cross_" + cross[q["id"]]} for q in json.loads((EXT / "cross_questions.json").read_text(encoding="utf-8"))]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    out = ROOT / "results" / args.out
    if out.exists():
        print(f"이미 결과가 있습니다(덮어쓰지 않음): {out}. (중단)")
        return 1
    spec = qd.load_prompt()
    llm = qd.build_llm()
    questions = load_questions()[: args.limit]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda q: {**q, **qd.decompose(llm, spec, q["question"])}, questions))
    out.mkdir(parents=True)
    with (out / "queries.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    usage = Counter()
    for r in results:
        for k, v in (r["usage"] or {}).items():
            if isinstance(v, int):
                usage[k] += v
    counts = Counter(len(r["queries"]) for r in results)
    summary = {"questions": len(results), "with_extra_queries": sum(1 for r in results if r["queries"]), "extra_query_count_distribution": dict(sorted(counts.items())),
               "errors": [{"question_id": r["question_id"], "error": r["error"]} for r in results if r["error"]],
               "mean_seconds": round(sum(r["seconds"] for r in results) / len(results), 2), "usage_total": dict(usage)}
    manifest = {"prompt": {"version": spec["version"], "sha256": spec["sha256"]}, "model_env": "OPENAI_MODEL", "reasoning_effort": "low", "max_completion_tokens": 2048,
                "seed": 20261111, "max_queries": qd.MAX_QUERIES, "concurrency": args.concurrency, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "errors"}, ensure_ascii=False), "| 오류", len(summary["errors"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
