"""STEP 12: 공식 비교 실행. 질문 60개 × RAG 방식 3개 = 답변 180개, 각 1회 생성 (DECISIONS D-045).

사용(서버를 먼저 띄운다):
    uv run uvicorn --factory src.api.main:build_default_app --port 8000
    uv run python scripts/preflight_official.py --write-manifest official-001     # 공식 실행 전에 한 번
    uv run python scripts/run_official.py --experiment-id official-001
    # 개발용 소량 확인: uv run python scripts/run_official.py --experiment-id official-dryrun-001 --track development --allow-dirty \\
    #                       --limit 2 --dataset dryrun-basic-v1

- 질문 파일(`*_questions.json`, 질문 번호와 질문뿐)만 읽는다. 정답 파일은 읽지 않는다.
- **실제 서버에 HTTP로 호출**하고, Langfuse Dataset/Experiments가 만든 Trace에 서비스가 `traceparent`로 이어 붙는다(D-029, D-031).
  방식마다 하나의 실험(run)이며 같은 질문(항목)끼리 Langfuse 비교 화면에서 나란히 볼 수 있다.
- **동시성 1**: 응답시간을 방식 사이에서 공정하게 비교하려고 요청을 한 번에 하나씩 보낸다(GPU 경합 방지).
- 공식 실행 전에 방식마다 1건의 **예열 요청**(실험 ID `{id}-warmup`, track development)을 보내 GPU 초기화 효과를 첫 방식에 몰리지 않게 하고,
  그 기록이 동결 값과 같은지 먼저 확인한다(다르면 본 실행 전에 중단).
- 서버 오류 요청도 오류로 남긴다(서버가 `runs.jsonl`에 기록). 결과 폴더에 이미 `runs.jsonl`이 있으면 덮어쓰지 않고 중단한다.
- 끝나면 180건이 (질문, 방식)마다 정확히 한 번씩 있는지, 버전이 동결과 같은지 확인해 `experiment_log.json`에 남긴다.
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

from src.config import load_env  # noqa: E402

load_env()

import httpx  # noqa: E402

from src.evaluation.compare import operational  # noqa: E402
from src.evaluation.official import (FREEZE_PATH, METHODS, check_runs_against_freeze, git_state, load_question_sets,  # noqa: E402
                                     sha256_file, verify_preflight)


def call_service(http, base_url: str, method: str, q: dict, *, experiment_id: str, track: str, traceparent: str | None, timeout: float,
                 client_log: list[dict]) -> dict:
    """서버에 질문 하나를 보낸다. 본문에는 질문 번호·질문·방식뿐이다(정답 정보 없음). 연결 실패는 오류 결과로 돌려주고 기록한다."""
    headers = {"X-Experiment-Id": experiment_id, "X-Track": track, "X-Run-Type": "general_eval"}
    if traceparent:
        headers["traceparent"] = traceparent
    body = {"question": q["question"], "rag_method": method, "question_id": q["question_id"]}
    t0 = time.perf_counter()
    try:
        r = http.post(f"{base_url}/rag/query", json=body, headers=headers, timeout=timeout)
        data = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        client_log.append({"question_id": q["question_id"], "rag_method": method, "error": type(exc).__name__, "message": str(exc)[:200]})
        return {"status": "client_error", "reason": type(exc).__name__, "answer": "", "citations": [], "http_status": None,
                "client_seconds": round(time.perf_counter() - t0, 2)}
    return {**data, "http_status": r.status_code, "x_request_id": r.headers.get("X-Request-Id"), "x_trace_id": r.headers.get("X-Trace-Id"),
            "client_seconds": round(time.perf_counter() - t0, 2)}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--dataset", default="basic-60q-v1", help="Langfuse 데이터셋 이름(질문 번호와 질문만 담김)")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--track", default="official", choices=["official", "development", "extension"])
    ap.add_argument("--limit", type=int, default=None, help="세트(development, additional)마다 앞의 N개만(소량 확인용)")
    ap.add_argument("--warmup", type=int, default=1, help="방식마다 예열 요청 수")
    ap.add_argument("--timeout", type=float, default=150.0)
    ap.add_argument("--wait-server", type=int, default=0, help="서버 /health가 응답할 때까지 최대 N초 기다림(기본 0, 바로 확인)")
    ap.add_argument("--no-langfuse", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true", help="작업 트리가 clean이 아니어도 진행(공식 track에서는 쓸 수 없음)")
    args = ap.parse_args()
    exp, out = args.experiment_id, ROOT / "results" / args.experiment_id

    if args.track == "official" and args.allow_dirty:
        print("공식(track=official) 실행은 작업 트리가 clean이어야 합니다. --allow-dirty를 쓸 수 없습니다. (중단)")
        return 1
    problems, _snap, git = verify_preflight(require_clean=not args.allow_dirty)
    if problems:
        print("사전 점검 실패:")
        for p in problems:
            print("  -", p)
        return 1
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if args.track == "official":
        mf = out / "manifest.json"
        if not mf.exists():
            print(f"manifest가 없습니다. 먼저 `preflight_official.py --write-manifest {exp}`를 실행하세요. (중단)")
            return 1
        m = json.loads(mf.read_text(encoding="utf-8"))
        if m.get("git_commit") != git["commit"] or m.get("freeze_sha256") != sha256_file(FREEZE_PATH):
            print("manifest의 git commit 또는 동결 파일 해시가 현재와 다릅니다(manifest 작성 뒤 코드나 동결이 바뀜). (중단)")
            return 1
    if (out / "runs.jsonl").exists():
        print(f"이미 실행 결과가 있습니다(덮어쓰지 않음): {out / 'runs.jsonl'}. (중단) 새 실험 ID를 쓰세요.")
        return 1

    http = httpx.Client()
    deadline = time.time() + args.wait_server  # 서버는 켜질 때 임베딩·리랭커 모델을 읽어서 시간이 걸린다
    while True:
        try:
            health = http.get(f"{args.base_url}/health", timeout=10).json()
            break
        except Exception as exc:  # noqa: BLE001
            if time.time() >= deadline:
                print(f"서버에 연결할 수 없습니다({type(exc).__name__}). 먼저 서버를 띄우세요. (중단)")
                return 1
            time.sleep(5)
    print(f"서버 {args.base_url}: {health}, git {git['commit'][:12]}, track={args.track}")

    questions = load_question_sets()
    if args.limit:
        questions = [q for s in ("development", "additional") for q in [x for x in questions if x["evaluation_set_type"] == s][:args.limit]]
    client_log: list[dict] = []

    # ---- 예열: 방식마다 1건 이상. 기록이 동결 값과 같은지 본 실행 전에 확인한다
    warm_id = f"{exp}-warmup"
    warm_ids: set[str] = set()
    for method in args.methods:
        for q in questions[:args.warmup]:
            res = call_service(http, args.base_url, method, q, experiment_id=warm_id, track="development", traceparent=None,
                               timeout=args.timeout, client_log=client_log)
            print(f"  예열 {method:13} {q['question_id']}: {res.get('status')} ({res.get('client_seconds')}초)")
            if res.get("x_request_id"):
                warm_ids.add(res["x_request_id"])
    warm_records = [r for r in read_jsonl(ROOT / "results" / warm_id / "runs.jsonl") if r.get("request_id") in warm_ids]
    drift = check_runs_against_freeze(warm_records, frozen)
    if len(warm_records) != len(warm_ids) or drift or client_log:
        print("예열 기록이 동결과 다르거나 오류가 있어 본 실행 전에 중단합니다:")
        for p in drift + [str(c) for c in client_log]:
            print("  -", p)
        return 1
    print("  예열 기록이 동결 값과 같습니다.")

    # ---- 본 실행
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state = {"consecutive_client_errors": 0, "aborted": False}
    runs_info: dict[str, dict] = {}

    def run_one(method: str, q: dict, traceparent: str | None) -> dict:
        if state["aborted"]:
            return {"status": "client_error", "reason": "aborted", "answer": "", "citations": []}
        res = call_service(http, args.base_url, method, q, experiment_id=exp, track=args.track, traceparent=traceparent,
                           timeout=args.timeout, client_log=client_log)
        state["consecutive_client_errors"] = state["consecutive_client_errors"] + 1 if res.get("status") == "client_error" else 0
        if state["consecutive_client_errors"] >= 3:
            state["aborted"] = True
            print("서버 연결이 3번 연속 실패해 이후 요청을 중단합니다.")
        return res

    if args.no_langfuse:
        for method in args.methods:
            for n, q in enumerate(questions, 1):
                res = run_one(method, q, None)
                print(f"  [{method} {n}/{len(questions)}] {q['question_id']:9} {res.get('status')} ({res.get('client_seconds')}초)")
            runs_info[method] = {"langfuse": False, "items": len(questions)}
    else:
        from src.observability.langfuse_client import get_langfuse

        lf = get_langfuse()
        if lf is None:
            print("Langfuse가 설정되지 않았습니다(--no-langfuse로 끌 수 있음). (중단)")
            return 1
        try:
            lf.get_dataset(args.dataset)
        except Exception:  # noqa: BLE001 - 없으면 만든다
            lf.create_dataset(name=args.dataset, description="공식 비교 질문(질문 번호와 질문만, 정답 없음)",
                              metadata={"source": "data/finance_basic/evaluation/*_questions.json", "freeze_sha256": sha256_file(FREEZE_PATH)})
        def with_retry(fn, attempts=4, delay=3.0):  # 같은 id로 다시 보내면 갱신되는 호출이라 일시적 네트워크 오류는 재시도해도 안전하다
            for i in range(attempts):
                try:
                    return fn()
                except Exception as exc:  # noqa: BLE001
                    if i == attempts - 1:
                        raise
                    print(f"  Langfuse 데이터셋 호출 재시도 {i + 1}/{attempts - 1} ({type(exc).__name__})")
                    time.sleep(delay * (i + 1))

        for q in questions:  # id를 `{데이터셋}:{질문 번호}`로 고정해 다시 실행해도 같은 항목으로 갱신된다(항목 id는 프로젝트 전체에서 유일해 데이터셋 이름을 붙임)
            with_retry(lambda q=q: lf.create_dataset_item(dataset_name=args.dataset, id=f"{args.dataset}:{q['question_id']}",
                                                          input={"question": q["question"], "question_id": q["question_id"]},
                                                          metadata={"evaluation_set_type": q["evaluation_set_type"]}))
        wanted = {q["question_id"] for q in questions}
        for method in args.methods:
            dataset = lf.get_dataset(args.dataset)

            def task(*, item, _m=method, **kwargs):
                inp = item.input if hasattr(item, "input") else item["input"]
                if inp["question_id"] not in wanted:  # 데이터셋에 남아 있는 다른 항목은 건너뜀(소량 확인용)
                    return {"status": "skipped", "answer": "", "citations": []}
                trace_id, span_id = lf.get_current_trace_id(), lf.get_current_observation_id()
                tp = f"00-{trace_id}-{span_id}-01" if trace_id and span_id else None
                q = next(x for x in questions if x["question_id"] == inp["question_id"])
                res = run_one(_m, q, tp)
                print(f"  [{_m}] {q['question_id']:9} {res.get('status')} ({res.get('client_seconds')}초)")
                return res

            result = dataset.run_experiment(name=f"{exp}-{method}", run_name=f"{exp}-{method}", task=task, max_concurrency=1,
                                            description=f"{exp} {method} ({args.track})",
                                            metadata={"experiment_id": exp, "rag_method": method, "track": args.track,
                                                      "freeze_sha256": sha256_file(FREEZE_PATH), "git_commit": git["commit"]})
            runs_info[method] = {"langfuse": True, "run_name": f"{exp}-{method}", "dataset_run_url": getattr(result, "dataset_run_url", None),
                                 "dataset_run_id": getattr(result, "dataset_run_id", None)}  # 건수는 runs.jsonl로 따로 검증한다
            lf.flush()
            if state["aborted"]:
                break
        lf.flush()

    # ---- 끝난 뒤 확인: (질문, 방식)마다 정확히 한 번, 버전은 동결과 같은가
    records = [r for r in read_jsonl(out / "runs.jsonl") if r.get("run_type") == "general_eval"]
    from collections import Counter

    counts = Counter((r["question_id"], r["rag_method"]) for r in records)
    expected = {(q["question_id"], m) for q in questions for m in args.methods}
    missing, duplicated = sorted(expected - set(counts)), sorted(k for k, v in counts.items() if v > 1)
    unexpected = sorted(set(counts) - expected)
    drift = check_runs_against_freeze(records, frozen)
    ops = {m: operational([r for r in records if r["rag_method"] == m]) for m in args.methods}
    log = {"experiment_id": exp, "track": args.track, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "git_commit": git["commit"], "dataset": None if args.no_langfuse else args.dataset, "runs": runs_info,
           "expected": len(expected), "found": len(records), "missing": missing, "duplicated": duplicated, "unexpected": unexpected,
           "client_errors": client_log, "version_problems": drift, "operational": ops, "aborted": state["aborted"]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "experiment_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n기대 {len(expected)}건, 기록 {len(records)}건, 누락 {len(missing)}, 중복 {len(duplicated)}, 클라이언트 오류 {len(client_log)}, 버전 문제 {len(drift)}")
    for m in args.methods:
        o = ops[m]
        print(f"  {m:13} 요청 {o['requests']} 정상 {o['ok']} 오류 {o['errors']} 평균 응답 {o['mean_total_ms'] and round(o['mean_total_ms'])}ms "
              f"입력 {o['input_tokens']} 출력 {o['output_tokens']} 비용 ${o['cost_usd']:.4f}")
    print(f"저장: {out / 'experiment_log.json'}")
    return 0 if not (missing or duplicated or unexpected or drift or state["aborted"]) else 2


if __name__ == "__main__":
    sys.exit(main())
