"""확장 브랜치 답변 단계: Intermediate 질문 78개를 확장 검색 구성의 서비스 파이프라인에 넣어 답변을 만든다 (Langfuse·Qdrant 없음).

사용:
    uv run python scripts/run_answers_ext.py --experiment-id ext-int-ans-base-001 --config base
    uv run python scripts/run_answers_ext.py --experiment-id ext-int-ans-v2b-001 --config v2b --limit 2     # 소량 점검

- 구성(`--config`): base = chunk-v1, v2b = 600자 청크(둘 다 e5-large + Hybrid + bge-reranker, 후보는 `--cands`, 기본 20,20,20,20). 최종 k=5, 문맥 4,000토큰,
  답변 프롬프트 answer_v2와 생성 설정은 Basic과 같다. 질문 분해·문서 라우팅은 `--queries-file` / `--routes-file`로 켠다(함께 쓰지 않음).
- 서비스에는 질문 번호와 질문 문자열만 보낸다(정답·근거·기대 행동 제외). 같은 요청 경로(`/rag/query`)로 TestClient가 호출하고 run 레코드는 서비스가 `results/{id}/runs.jsonl`에 쓴다.
- 질문은 제공 질문 60개(development 30 + additional 30)와 자체 질문 18개(동결)다. 결과 폴더가 이미 있으면 덮어쓰지 않고 중단한다.
- 결과: runs.jsonl, manifest.json(구성·프롬프트 해시·질문 수), summary.json(상태 분포, 오류, 평균 응답시간, 토큰, 비용).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_env  # noqa: E402

load_env()

from fastapi.testclient import TestClient  # noqa: E402

from src.api.main_ext import build_ext_app  # noqa: E402
from src.evaluation.compare import operational  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from decompose_queries_ext import load_questions  # noqa: E402

CONFIGS = {
    "base": {"processed": ROOT / "data" / "processed_intermediate", "index": "intermediate__chunk-v1__e5large-v1", "version": "chunk-v1"},
    "v2b": {"processed": ROOT / "data" / "processed_intermediate_v2b", "index": "intermediate__chunk-v2b-max600__e5large-v1", "version": "chunk-v2b-max600"},
}


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def lookup_from(name: str | None, key: str) -> dict | None:
    if not name:
        return None
    rows = [json.loads(l) for l in (ROOT / "results" / name / f"{'queries' if key == 'queries' else 'routes'}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["question"]: r["queries" if key == "queries" else "doc_ids"] for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--cands", default="20,20,20,20", help="dense,bm25,fused,rerank 후보 수")
    ap.add_argument("--rag-method", default="hybrid_rerank", choices=["dense", "hybrid", "hybrid_rerank"])
    ap.add_argument("--final-k", type=int, help="최종 청크 수(기본 5, Basic과 같음). 문맥 4,000토큰 한도는 그대로 적용된다")
    ap.add_argument("--queries-file")
    ap.add_argument("--routes-file")
    ap.add_argument("--sets", nargs="+", choices=["development", "additional", "ext_dev", "ext_hold"], help="이 세트의 질문만 실행(기본: 78개 전부). 예: --sets development ext_dev = 판단용 tune 39개")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    out = ROOT / "results" / args.experiment_id
    if out.exists():
        print(f"이미 결과가 있습니다(덮어쓰지 않음): {out}. (중단)")
        return 1
    cfg = CONFIGS[args.config]
    cands = tuple(int(x) for x in args.cands.split(","))
    app, pipeline = build_ext_app(processed_dir=cfg["processed"], index_name=cfg["index"], chunking_version=cfg["version"], cands=cands, final_k=args.final_k,  # type: ignore[arg-type]
                                  queries_lookup=lookup_from(args.queries_file, "queries"), routes_lookup=lookup_from(args.routes_file, "routes"))
    client = TestClient(app, raise_server_exceptions=False)
    headers = {"X-Experiment-Id": args.experiment_id, "X-Track": "extension", "X-Run-Type": "general_eval"}
    questions = [q for q in load_questions() if not args.sets or q["set"] in args.sets][: args.limit]
    results = []
    for q in questions:
        r = client.post("/rag/query", json={"question": q["question"], "rag_method": args.rag_method, "question_id": q["question_id"]}, headers=headers)
        body = r.json()
        results.append((q, r.status_code, body))
        print(f"  {q['question_id']:11} HTTP {r.status_code} {body['status']:19} 인용 {len(body.get('citations') or [])}")
    recs = [json.loads(l) for l in (out / "runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ops = operational(recs)
    summary = {"questions": len(results), "records": len(recs), "http_status": dict(Counter(s for _, s, _ in results)),
               "response_status": dict(Counter(b["status"] for _, _, b in results)), "errors": [{"question_id": q["question_id"], "reason": b.get("reason")} for q, s, b in results if b["status"] == "error"],
               "operational": ops}
    manifest = {"experiment_id": args.experiment_id, "track": "extension", "config": args.config, "chunking_version": cfg["version"], "index": cfg["index"], "cands": list(cands),
                "rag_method": args.rag_method, "queries_file": args.queries_file, "routes_file": args.routes_file, "final_k": pipeline.settings.final_k,
                "prompt_version": pipeline.generator.prompt_version, "prompt_sha256": pipeline.generator.prompt_sha256, "questions": len(questions), "sets": args.sets, "langfuse": False, "qdrant": False,
                "note": "확장 브랜치 답변 단계 평가용. 사람 확인 없이 자동 평가만 쓴다. 자체 질문은 AI 작성(사람 검증 없음)이며 공식 점수에 합산하지 않는다.",
                "git_commit": git_commit(), "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n요약:", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
