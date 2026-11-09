"""STEP 10: 대표 요청을 실제 서버 구성(Qdrant·e5·Reranker·gpt-6-luna)으로 보내고 Langfuse 기록을 로컬 기록과 대조한다.

사용: uv run python scripts/verify_langfuse.py [실험ID]             (기본 langfuse-verify-001, 요청 7건을 새로 보낸다)
      uv run python scripts/verify_langfuse.py --recheck [실험ID]  (이미 보낸 요청의 Langfuse 기록을 다시 점검만 한다. 요청을
                                                                    보내지 않고 report.json은 두며 report_recheck.json에 쓴다)

- 요청 7건(Langfuse에 약 60유닛): 세 방식의 정상 답변, 되묻기, 모델 호출 시간 초과(재시도 3회), 빈 질문(422), 필드 누락(422).
  오류 요청은 `run_type = mock_error`로 기록해 일반 평가 지표와 섞이지 않게 한다.
- 로컬 run 레코드는 results/{실험ID}/runs.jsonl에, 점검 결과는 report.json에 저장한다. 폴더가 이미 있으면 중단한다.
- 점검 로직은 src/observability/trace_checks.py(단위 테스트됨). 비밀 값은 출력하지 않고 개수만 보고한다.
- 평가 LLM 호출(scoring)과 서비스 호출의 구분은 STEP 11에서 평가가 생긴 뒤 확인한다.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_env  # noqa: E402

load_env()

from fastapi.testclient import TestClient  # noqa: E402
from langfuse.langchain import CallbackHandler  # noqa: E402

from src.api.app import create_app  # noqa: E402
from src.api.pipeline import PipelineSettings, RagPipeline  # noqa: E402
from src.api.recording import RunRecorder  # noqa: E402
from src.generation.generator import LangChainGenerator  # noqa: E402
from src.generation.pricing import load_pricing  # noqa: E402
from src.observability.langfuse_client import get_langfuse  # noqa: E402
from src.observability.trace_checks import check_batch, normalize  # noqa: E402
from src.retrieval.dense import DenseRetriever  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402
from src.retrieval.rerank import CrossEncoderReranker, RerankRetriever  # noqa: E402
from src.retrieval.sparse import SparseRetriever  # noqa: E402

DEFAULT_ID = "langfuse-verify-001"


def verify_and_report(lf, out: Path, experiment: str, sent: list[dict], filename: str) -> int:
    records = [json.loads(line) for line in (out / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    by_trace: dict[str, list] = {}
    for rec in records:
        tid = rec.get("trace_id")
        if not tid:
            continue
        want_gens = len(rec.get("llm_calls") or [])
        for _ in range(24):  # 도착을 기다린다(최대 약 2분)
            obs = lf.api.observations.get_many(trace_id=tid, limit=100, fields="core,basic,io,model,usage,metadata").data
            if sum(o.type == "GENERATION" for o in obs) >= want_gens and any(o.name == "rag-query" and o.output for o in obs):
                break
            time.sleep(5)
        by_trace[tid] = obs
    report = check_batch(records, by_trace)

    for rec in records:
        r = report["per_request"][rec["request_id"]]
        bad = [(n, d) for n, ok, d in r["checks"] if not ok]
        print(f"\n[{rec['rag_method'] or '-'} / HTTP {rec['http_status']} / {rec['response']['status']}] 요청 {rec['request_id'][:8]}… "
              f"Trace {str(rec.get('trace_id'))[:8]}… 검사 {len(r['checks'])}개 중 통과 {len(r['checks']) - len(bad)}개")
        for n, d in bad:
            print(f"   실패: {n} — {d}")
    print("\n요청 사이 검사:", [(n, ok, d) for n, ok, d in report["global"]])
    print("전체 통과:", report["passed"])
    rep = next((r for r in records if r["rag_method"] == "hybrid_rerank" and r["http_status"] == 200), None)
    if rep:
        print("대표 요청(Hybrid+Rerank) 추적 URL:", lf.get_trace_url(trace_id=rep["trace_id"]))
    (out / filename).write_text(json.dumps({
        "experiment_id": experiment, "track": "development", "step": "10", "requests": sent,
        "result": {rid: {**v, "checks": [{"name": n, "passed": ok, "detail": d} for n, ok, d in v["checks"]]}
                   for rid, v in report["per_request"].items()},
        "global": [{"name": n, "passed": ok, "detail": d} for n, ok, d in report["global"]], "passed": report["passed"],
        "representative_trace_id": rep["trace_id"] if rep else None,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return 0 if report["passed"] else 2


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    recheck = "--recheck" in sys.argv
    experiment = args[0] if args else DEFAULT_ID
    out = ROOT / "results" / experiment
    lf = get_langfuse()
    if lf is None:
        print("Langfuse 설정이 없습니다. (중단)")
        return 1
    if recheck:
        if not (out / "runs.jsonl").exists():
            print(f"점검할 기록이 없습니다: {out}. (중단)")
            return 1
        if (out / "report_recheck.json").exists():
            print("report_recheck.json이 이미 있습니다(덮어쓰지 않음). (중단)")
            return 1
        return verify_and_report(lf, out, experiment, [], "report_recheck.json")
    if out.exists():
        print(f"이미 결과가 있습니다(덮어쓰지 않음): {out}. (중단)")
        return 1

    dense = DenseRetriever(langfuse=lf)
    hybrid = HybridRetriever(dense, SparseRetriever(langfuse=lf), langfuse=lf)
    retrievers = {"dense": dense, "hybrid": hybrid,
                  "hybrid_rerank": RerankRetriever(hybrid, CrossEncoderReranker(), langfuse=lf)}
    factory = lambda: [CallbackHandler()]  # noqa: E731
    recorder = RunRecorder(ROOT / "results")

    def app_for(settings: PipelineSettings):
        pipe = RagPipeline(dense, LangChainGenerator(settings, callbacks_factory=factory), settings,
                           pricing=load_pricing(), retrievers=retrievers, langfuse=lf)
        return TestClient(create_app(pipe, langfuse=lf, recorder=recorder), raise_server_exceptions=False)

    normal = app_for(PipelineSettings())
    # 모델 호출이 반드시 시간 초과되도록 timeout을 0.001초로 둔 같은 구성(재시도 2회 → 호출 3회). 오류·재시도 기록 확인용
    timeout = app_for(PipelineSettings(model_timeout_seconds=0.001, retry_delays_seconds=(0.2, 0.2)))

    def headers(run_type):
        return {"X-Experiment-Id": experiment, "X-Track": "development", "X-Run-Type": run_type}

    plan = [
        ("정상: Dense", normal, {"question": "2026년 상반기 영업이익은 얼마인가요?", "rag_method": "dense"}, "general_eval"),
        ("정상: Hybrid", normal, {"question": "루미센서 E는 어떤 제품인가요?", "rag_method": "hybrid"}, "general_eval"),
        ("정상: Hybrid+Rerank", normal, {"question": "리브필름 기본형의 재생 원료 비율은?", "rag_method": "hybrid_rerank"}, "general_eval"),
        ("되묻기", normal, {"question": "이번 분기 매출이 얼마인가요?", "rag_method": "dense"}, "general_eval"),
        ("오류: 모델 시간 초과(재시도 3회)", timeout, {"question": "영업이익은 얼마인가요?", "rag_method": "dense"}, "mock_error"),
        ("오류: 빈 질문(422)", normal, {"question": "   ", "rag_method": "dense"}, "mock_error"),
        ("오류: 필드 누락(422)", normal, {"question": "질문만 있음"}, "mock_error"),
    ]
    sent = []
    for label, client, body, run_type in plan:
        r = client.post("/rag/query", json=body, headers=headers(run_type))
        sent.append({"label": label, "request_id": r.json().get("request_id"), "http_status": r.status_code,
                     "status": r.json().get("status"), "reason": r.json().get("reason"), "trace_id": r.headers.get("x-trace-id")})
        print(f"{label:34} HTTP {r.status_code} {r.json().get('status')} ({r.json().get('reason')[:40]})")
    lf.flush()
    return verify_and_report(lf, out, experiment, sent, "report.json")


if __name__ == "__main__":
    sys.exit(main())
