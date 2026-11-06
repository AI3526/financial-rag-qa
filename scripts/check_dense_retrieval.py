"""STEP 4: 질문 유형별 Dense 검색을 실행해 결과를 results/dense-check-001/ 에 저장하고 Langfuse에도 기록한다.

사용: uv run python scripts/check_dense_retrieval.py [--no-langfuse]

- 질문은 평가 데이터가 아니라 문서의 이름·용어로 직접 만든 것이다(정답·근거 원문 미사용).
- 결과 폴더가 이미 있으면 덮어쓰지 않고 중단한다. 성능 지표가 아니라 동작 확인 기록이다(track = development).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_env  # noqa: E402

load_env()

from src.observability.ingestion_trace import RetrievalCheckTracer  # noqa: E402
from src.observability.langfuse_client import get_langfuse  # noqa: E402
from src.retrieval.dense import DenseRetriever, RetrievalConfig  # noqa: E402

EXPERIMENT_ID = "dense-check-001"
OUT = ROOT / "results" / EXPERIMENT_ID

QUERIES = [
    ("회사명", "해온소재는 어떤 회사인가요?"),
    ("제품명", "루미센서 E는 어떤 제품인가요?"),
    ("제품명", "리브필름 기본형의 재생 원료 비율은?"),
    ("기간", "2026년 상반기 실적을 알려 주세요."),
    ("수치", "2026년 상반기 영업이익은 얼마인가요?"),
    ("사업부문", "장비사업과 운영서비스사업은 어떻게 구분되나요?"),
    ("근거 부족", "해온솔루션의 2027년 1분기 확정 매출을 알려 주세요."),
    ("무관한 질문", "오늘 저녁 메뉴로 김치찌개가 좋을까요?"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-langfuse", action="store_true")
    args = parser.parse_args()
    if OUT.exists():
        print(f"이미 결과가 있습니다(덮어쓰지 않음): {OUT}. (중단)")
        return 1

    cfg = RetrievalConfig()
    langfuse = None if args.no_langfuse else get_langfuse()
    retriever = DenseRetriever(cfg, langfuse=langfuse)
    meta = {"step": "4", "experiment_id": EXPERIMENT_ID, "collection": cfg.collection, "k": cfg.k,
            "queries": len(QUERIES)}
    records = []
    with RetrievalCheckTracer(langfuse, meta) as tracer:
        for kind, question in QUERIES:
            res = retriever.retrieve(question)
            records.append({"kind": kind, "question": question, "k": res.k, "durations_ms": res.durations_ms,
                            "chunks": res.chunks})
            top = res.chunks[0]
            print(f"[{kind}] {question}\n   1위 {top['score']:.4f} {top['file_name']} p.{top['page']} "
                  f"({res.durations_ms['total']}ms)")
    if langfuse is not None:
        print(f"Langfuse: {tracer.trace_url()}")

    OUT.mkdir(parents=True)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (OUT / "manifest.json").write_text(json.dumps({
        "experiment_id": EXPERIMENT_ID, "track": "development", "step": "4", "retrieval": "dense",
        "collection": cfg.collection, "embedding_model": cfg.spec.hf_id, "k": cfg.k,
        "purpose": "동작 확인(성능 지표 아님)", "langfuse_trace_id": tracer.trace_id, "created_at": created,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "retrievals.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"저장: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
