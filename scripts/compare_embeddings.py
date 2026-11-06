"""STEP 3: 임베딩 후보(e5, BGE-M3) 1회 소규모 확인 (06_MODEL_SELECTION.md §11.2, DECISIONS.md D-026).

사용: uv run python scripts/compare_embeddings.py [--no-langfuse]

- development 30문항, Dense Top-5, Hit@5/Recall@5(book_id 단위, 공식)와 (book_id, page) 단위(참고).
- 두 컬렉션이 먼저 만들어져 있어야 한다(build_index.py). 질문 텍스트만 임베딩하고 정답·근거는 검색에 쓰지 않는다.
- 선정 규칙(D-026)을 코드로 적용해 승자를 계산해 보여 준다. 동결(승인)은 사용자가 한다.
- 결과: results/dev-embed-select-001/ (이미 있으면 중단)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.retrieval_metrics import mean_ignoring_none, question_metrics  # noqa: E402
from src.config import load_env  # noqa: E402

load_env()  # huggingface_hub보다 먼저: 빈 HF_HOME 등을 제거

from src.index import qdrant_store as qs  # noqa: E402
from src.index.embedding import CANDIDATES, Embedder, collection_name  # noqa: E402
from src.observability.ingestion_trace import ComparisonTracer  # noqa: E402
from src.observability.langfuse_client import get_langfuse  # noqa: E402

EXPERIMENT_ID = "dev-embed-select-001"
K = 5
DEV = ROOT / "data" / "finance_basic" / "evaluation" / "development.json"
OUT = ROOT / "results" / EXPERIMENT_ID
PROCESSED = ROOT / "data" / "processed"


def evaluate_model(spec, questions, client, chunking_version, dataset_id) -> dict:
    name = collection_name(dataset_id, chunking_version, spec)
    manifest = json.loads((PROCESSED / f"index_manifest__{spec.version_tag}.json").read_text(encoding="utf-8"))
    embedder = Embedder(spec)
    texts = [q["question"] for q in questions]  # 검색 입력은 질문 문자열뿐
    t = time.perf_counter()
    vectors = [embedder.embed_queries([x])[0] for x in texts]  # 질문 1개씩: 실제 서비스 호출 방식과 같다
    query_seconds = time.perf_counter() - t
    rows = []
    for q, vec in zip(questions, vectors, strict=True):
        hits = qs.search(client, name, vec, k=K)
        m = question_metrics(hits, q["evidence_groups"])
        rows.append({
            "question_id": q["id"], "expected_action": q["expected_action"], **m,
            "retrieved": [{"chunk_id": h["chunk_id"], "book_id": h["book_id"], "page": h["page"],
                           "score": round(h["score"], 4)} for h in hits],
        })
    index_embed_seconds = manifest["indexing_seconds"]["embedding"]
    summary = {
        "model": spec.key, "collection": name, "embedding_revision": embedder.revision,
        "questions": len(rows), "evaluated_questions": sum(r["hit"] is not None for r in rows),
        "hit_at_5": mean_ignoring_none([r["hit"] for r in rows]),
        "recall_at_5": mean_ignoring_none([r["recall"] for r in rows]),
        "page_hit_at_5": mean_ignoring_none([r["page_hit"] for r in rows]),
        "page_recall_at_5": mean_ignoring_none([r["page_recall"] for r in rows]),
        "index_embed_seconds": index_embed_seconds, "query_embed_seconds": round(query_seconds, 3),
        "time_score_seconds": round(index_embed_seconds + query_seconds, 3),
    }
    return {"summary": summary, "rows": rows}


def pick_winner(results: dict[str, dict]) -> tuple[str, str]:
    """D-026 규칙: Recall@5 → Hit@5 → 시간(짧은 쪽). 동점은 소수 9자리 기준."""
    a, b = (results[k]["summary"] for k in results)
    for field, higher_better, label in (("recall_at_5", True, "Recall@5"), ("hit_at_5", True, "Hit@5"),
                                        ("time_score_seconds", False, "시간")):
        x, y = round(a[field], 9), round(b[field], 9)
        if x != y:
            win = a if (x > y) == higher_better else b
            return win["model"], label
    return a["model"], "완전 동점(첫 후보)"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-langfuse", action="store_true")
    args = parser.parse_args()
    if OUT.exists():
        print(f"이미 결과가 있습니다(덮어쓰지 않음): {OUT}. (중단)")
        return 1

    questions = json.loads(DEV.read_text(encoding="utf-8"))
    chunking = json.loads((PROCESSED / "chunking_manifest.json").read_text(encoding="utf-8"))
    client = qs.get_client()
    langfuse = None if args.no_langfuse else get_langfuse()
    meta = {"step": "3", "experiment_id": EXPERIMENT_ID, "k": K, "question_set": "development",
            "questions": len(questions)}
    results: dict[str, dict] = {}
    with ComparisonTracer(langfuse, meta) as tracer:
        for key, spec in CANDIDATES.items():
            def work(s=spec) -> dict:
                r = evaluate_model(s, questions, client, chunking["chunking_version"], chunking["dataset_id"])
                results[s.key] = r
                return {"summary": r["summary"],
                        "per_question": [{k: row[k] for k in ("question_id", "hit", "recall", "page_hit", "page_recall")}
                                         | {"retrieved": [(h["book_id"], h["page"]) for h in row["retrieved"]]}
                                         for row in r["rows"]]}

            tracer.record(f"retrieve-eval {key}", {"model": spec.hf_id, "k": K}, {"model": key}, work)
            s = results[key]["summary"]
            print(f"{key:6} Hit@5={s['hit_at_5']:.4f} Recall@5={s['recall_at_5']:.4f} | 페이지 단위 "
                  f"Hit={s['page_hit_at_5']:.4f} Recall={s['page_recall_at_5']:.4f} | 시간 {s['time_score_seconds']}s "
                  f"(색인 {s['index_embed_seconds']}s + 질문 {s['query_embed_seconds']}s)")
    if langfuse is not None:
        print(f"Langfuse: 모델 Span {tracer.documents_recorded}개 기록\n  {tracer.trace_url()}")

    winner, reason = pick_winner(results)
    print(f"규칙 적용 결과: 승자 {winner} (결정 기준: {reason})")

    OUT.mkdir(parents=True)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (OUT / "manifest.json").write_text(json.dumps({
        "experiment_id": EXPERIMENT_ID, "track": "development", "step": "3", "k": K,
        "question_set": "development", "questions": len(questions), "retrieval": "dense",
        "selection_rule": "DECISIONS.md D-026", "chunking_version": chunking["chunking_version"],
        "candidates": {k: {"collection": r["summary"]["collection"], "revision": r["summary"]["embedding_revision"]}
                       for k, r in results.items()},
        "langfuse_trace_id": tracer.trace_id, "created_at": created,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({
        "models": {k: r["summary"] for k, r in results.items()},
        "rule_result": {"winner": winner, "decided_by": reason, "frozen": False},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "per_question.jsonl").open("w", encoding="utf-8") as fh:
        for key, r in results.items():
            for row in r["rows"]:
                fh.write(json.dumps({"model": key, **row}, ensure_ascii=False) + "\n")
    print(f"저장: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
