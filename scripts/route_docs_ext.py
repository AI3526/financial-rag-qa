"""확장 브랜치 방법 A: 질문마다 읽어야 할 문서를 골라 저장한다 (모델 호출, Langfuse 미사용).

사용: uv run python scripts/route_docs_ext.py --out ext-int-s6-routes-v1

- 대상: Intermediate 제공 질문(development 30 + additional 30)과 자체 질문(`evaluation_ext`, 동결) 전체. **질문 번호와 질문 문자열, 문서 목록(번호·제목)만** 모델에 보낸다.
- 프롬프트: `prompts/doc_route_v1.json`(사용자 승인 문안). 질문당 1회 호출하고 결과를 저장해 평가에서 재사용한다.
- 결과: `results/{out}/routes.jsonl`(질문별 고른 문서·버려진 잘못된 번호 수·원문 출력·사용량·시간·오류), `summary.json`, `manifest.json`(프롬프트 해시, 문서 목록 해시와 전문).
- 결과 폴더가 이미 있으면 덮어쓰지 않고 중단한다.
"""
from __future__ import annotations

import argparse
import hashlib
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

from src.retrieval import doc_route as dr  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from decompose_queries_ext import load_questions  # noqa: E402


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
    spec = dr.load_prompt()
    pages = [json.loads(l) for l in (ROOT / "data" / "processed_intermediate" / "pages.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    catalog = dr.build_catalog(pages)
    text = dr.catalog_text(catalog)
    valid = {b for b, _ in catalog}
    llm = dr.build_llm()
    questions = load_questions()[: args.limit]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda q: {**q, **dr.route(llm, spec, text, valid, q["question"])}, questions))
    out.mkdir(parents=True)
    with (out / "routes.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    usage = Counter()
    for r in results:
        for k, v in (r["usage"] or {}).items():
            if isinstance(v, int):
                usage[k] += v
    summary = {"questions": len(results), "with_docs": sum(1 for r in results if r["doc_ids"]), "doc_count_distribution": dict(sorted(Counter(len(r["doc_ids"]) for r in results).items())),
               "invalid_ids_dropped": sum(r["invalid_ids"] for r in results), "errors": [{"question_id": r["question_id"], "error": r["error"]} for r in results if r["error"]],
               "mean_seconds": round(sum(r["seconds"] for r in results) / len(results), 2), "usage_total": dict(usage)}
    manifest = {"prompt": {"version": spec["version"], "sha256": spec["sha256"]}, "catalog": {"documents": len(catalog), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "text": text},
                "model_env": "OPENAI_MODEL", "reasoning_effort": "low", "max_completion_tokens": 2048, "seed": 20261111, "max_docs": dr.MAX_DOCS,
                "concurrency": args.concurrency, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "errors"}, ensure_ascii=False), "| 오류", len(summary["errors"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
