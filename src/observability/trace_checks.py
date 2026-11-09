"""Langfuse에 남은 요청 Trace를 로컬 run 레코드와 대조하는 검사 (STEP 10, 05_OBSERVABILITY_PLAN.md §15).

`check_trace(record, observations)`는 요청 1건의 로컬 기록(`runs.jsonl`의 한 행)과 Langfuse에서 읽은 관측(Observation) 목록을
비교해 항목별 (이름, 통과 여부, 설명)을 돌려준다. 확인하는 것:
- 요청 하나에 루트 Trace 하나, 모든 관측이 그 아래에 연결됨, Request ID·Trace ID 연결
- 필수 식별 정보(metadata)와 버전
- 방식별 검색 Span, 검색 결과와 최종 문맥의 구분(최종 문맥이 모델 입력과 일치)
- 모델 입력·출력·사용량, 재시도 횟수, 출처 변환과 상태
- 오류의 유형과 발생 단계, 실행시간
- API 키 등 비밀 값이 기록에 남지 않았는지(값은 출력하지 않고 개수만 보고한다)
평가용 LLM 호출(scoring)과의 구분은 STEP 11에서 평가가 생긴 뒤 확인한다(여기서는 서비스 호출만 다룬다).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

REQUIRED_METADATA = ("request_id", "run_type", "track", "rag_method", "config_version", "dataset_id", "dataset_version",
                     "prompt_version")
EXPECTED_RETRIEVAL_SPANS = {
    "dense": ["dense-retrieve"],
    "hybrid": ["hybrid-retrieve", "dense-retrieve", "sparse-retrieve"],
    "hybrid_rerank": ["rerank-retrieve", "hybrid-retrieve", "dense-retrieve", "sparse-retrieve"],
}
_SECRET_PATTERNS = [re.compile(p) for p in (r"sk-[A-Za-z0-9_\-]{12,}", r"sk-lf-[A-Za-z0-9\-]{8,}")]
# 비밀 값: 기록에 하나라도 있으면 실패. LANGFUSE_PUBLIC_KEY는 이름 그대로 공개 식별자이고 Langfuse SDK가 모든 관측의
# metadata(`scope.attributes.public_key`)에 스스로 넣는다(STEP 10에서 확인). 그 위치에 있는 것만 허용하고 따로 보고한다.
SECRET_ENV_NAMES = ("OPENAI_API_KEY", "LANGFUSE_SECRET_KEY", "QDRANT_API_KEY")
PUBLIC_KEY_ENV = "LANGFUSE_PUBLIC_KEY"
SDK_PUBLIC_KEY_PATH = "metadata.scope.attributes.public_key"


def _parse(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def normalize(obs: Any) -> dict:
    """Langfuse 관측 객체(또는 dict)를 검사용 dict로 바꾼다. input·output·metadata의 JSON 문자열은 풀어 준다."""
    get = obs.get if isinstance(obs, dict) else (lambda k, d=None: getattr(obs, k, d))
    return {"id": get("id"), "name": get("name"), "type": get("type"), "parent_id": get("parent_observation_id"),
            "trace_id": get("trace_id"), "level": get("level"), "status_message": get("status_message"),
            "input": _parse(get("input")), "output": _parse(get("output")), "metadata": _parse(get("metadata")) or {},
            "model": get("model"), "usage": get("usage_details") or {}}


def find_secrets(observations: list[dict], env: dict | None = None) -> int:
    """기록 전체(입력·출력·metadata·상태 메시지)에서 비밀 값이나 키 형태가 나온 횟수. 값 자체는 돌려주지 않는다.

    같은 자리에서 환경 변수 값과 키 형태가 함께 걸리면 한 번으로 센다(겹치는 일치는 합친다).
    """
    env = os.environ if env is None else env
    secrets = [env[n] for n in SECRET_ENV_NAMES if env.get(n) and len(env[n]) >= 8]
    blob = json.dumps([[o["input"], o["output"], o["metadata"], o["status_message"]] for o in observations], ensure_ascii=False)
    spans: list[tuple[int, int]] = []
    for secret in secrets:
        start = blob.find(secret)
        while start != -1:
            spans.append((start, start + len(secret)))
            start = blob.find(secret, start + len(secret))
    for pattern in _SECRET_PATTERNS:
        spans += [m.span() for m in pattern.finditer(blob)]
    merged = 0
    last_end = -1
    for s_, e_ in sorted(spans):
        if s_ >= last_end:  # 앞의 일치와 겹치지 않으면 새 발견
            merged += 1
        last_end = max(last_end, e_)
    return merged


def _walk(x: Any, path: str = ""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(x, list):
        for v in x:
            yield from _walk(v, f"{path}[]")
    else:
        yield path, x


def public_key_locations(observations: list[dict], env: dict | None = None) -> dict[str, int]:
    """Langfuse 공개 키가 나온 위치(경로)별 횟수. 값은 돌려주지 않는다."""
    env = os.environ if env is None else env
    key = env.get(PUBLIC_KEY_ENV)
    found: dict[str, int] = {}
    if not key or len(key) < 8:
        return found
    for o in observations:
        for field in ("input", "output", "metadata", "status_message"):
            for path, val in _walk(o[field], field):
                if key in json.dumps(val, ensure_ascii=False):
                    found[path] = found.get(path, 0) + 1
    return found


def _root(obs: list[dict]) -> dict | None:
    ids = {o["id"] for o in obs}
    roots = [o for o in obs if o["name"] == "rag-query"]
    return roots[0] if len(roots) == 1 else None


def _descends_from(obs: list[dict], root_id: str) -> bool:
    by = {o["id"]: o for o in obs}
    for o in obs:
        cur, seen = o, set()
        while cur["id"] != root_id:
            pid = cur["parent_id"]
            if pid is None or pid in seen or pid not in by:
                return False  # 루트 밖으로 끊어졌다
            seen.add(cur["id"])
            cur = by[pid]
    return True


def check_trace(record: dict, observations: list[Any], env: dict | None = None) -> list[tuple[str, bool, str]]:
    obs = [normalize(o) for o in observations]
    out: list[tuple[str, bool, str]] = []
    add = lambda name, ok, detail="": out.append((name, bool(ok), detail))  # noqa: E731

    root = _root(obs)
    roots = [o for o in obs if o["name"] == "rag-query"]
    add("루트 Trace 1개(rag-query)", len(roots) == 1, f"rag-query {len(roots)}개")
    if root is None:
        return out + [("이후 검사 불가", False, "루트가 없거나 둘 이상")]
    rid_ok = [o for o in obs if o["trace_id"] and record.get("trace_id") and o["trace_id"] != record["trace_id"]]
    add("Trace ID 연결(로컬 trace_id == Langfuse)", not rid_ok and bool(record.get("trace_id")), f"불일치 관측 {len(rid_ok)}개")
    add("모든 관측이 루트 아래에 연결", _descends_from(obs, root["id"]), f"관측 {len(obs)}개")
    md = root["metadata"]
    add("Request ID 연결(metadata.request_id)", md.get("request_id") == record["request_id"],
        f"langfuse={str(md.get('request_id'))[:8]} 로컬={record['request_id'][:8]}")
    missing = [k for k in REQUIRED_METADATA if not md.get(k)]
    add("필수 식별 정보(metadata)", not missing, f"누락 {missing}" if missing else "모두 있음")
    add("식별 정보 값이 로컬 기록과 일치", all(str(md.get(k)) == str(record.get(k)) for k in ("run_type", "track", "rag_method")),
        f"run_type={md.get('run_type')} track={md.get('track')} rag_method={md.get('rag_method')}")
    add("config_version 일치", md.get("config_version") == (record.get("versions") or {}).get("config_version"),
        str(md.get("config_version")))
    add("로컬 기록의 langfuse_recorded", (record.get("observability") or {}).get("langfuse_recorded") is True, "")

    err = record.get("error")
    names = [o["name"] for o in obs]
    stage = (err or {}).get("error_stage")
    reached_retrieval = err is None or stage not in ("validate",)
    method = record.get("rag_method")
    if reached_retrieval and method in EXPECTED_RETRIEVAL_SPANS and not (err and stage == "retrieval"):
        want = EXPECTED_RETRIEVAL_SPANS[method]
        miss = [n for n in want if n not in names]
        add(f"검색 Span({method})", not miss, f"누락 {miss}" if miss else " > ".join(want))
    root_out = root["output"] if isinstance(root["output"], dict) else {}

    if err is None or stage == "generation":
        cc = next((o for o in obs if o["name"] == "context-construction"), None)
        add("최종 문맥 Span(context-construction)", cc is not None, "")
        if cc is not None and isinstance(cc["output"], dict):
            local_final = [c["chunk_id"] for c in (record.get("final_context") or {}).get("chunks", [])]
            lf_final = [c["chunk_id"] for c in cc["output"].get("final_chunks", [])]
            add("최종 문맥 청크가 로컬 기록과 일치", lf_final == local_final, f"{len(lf_final)}개")
            retrieved = [c["chunk_id"] for c in (cc["input"] or {}).get("retrieved", [])]
            dropped = cc["output"].get("dropped_chunk_ids", [])
            add("검색 결과와 최종 문맥의 구분(검색 − 제외 = 최종)", [c for c in retrieved if c not in dropped] == lf_final,
                f"검색 {len(retrieved)}개, 제외 {len(dropped)}개, 최종 {len(lf_final)}개")
        gens = [o for o in obs if o["type"] == "GENERATION"]
        calls = record.get("llm_calls") or []
        add("모델 호출(answer-generation) 수 == 로컬 llm_calls 수(재시도 포함)", len(gens) == len(calls) and all(o["name"] == "answer-generation" for o in gens),
            f"Langfuse {len(gens)}개, 로컬 {len(calls)}개")
        if gens:
            add("모델 입력 기록(system·user 메시지)", all(g["input"] for g in gens), "")
            add("모델명 기록", all(g["model"] for g in gens), str({g["model"] for g in gens}))
            if err is None:
                add("모델 출력 기록", any(g["output"] for g in gens), "")
                ok_call = next((c for c in calls if c.get("status") == "ok"), None)
                ok_gen = next((g for g in gens if g["output"] and g["level"] != "ERROR"), None)
                if ok_call and ok_gen:
                    u = ok_gen["usage"]
                    total_in = (u.get("input") or 0) + (u.get("input_cache_read") or 0) + (u.get("input_cache_creation") or 0)
                    add("사용량(입력 토큰 합 == 로컬 input_tokens)", total_in == ok_call.get("input_tokens"), f"Langfuse {total_in} 로컬 {ok_call.get('input_tokens')}")
                    add("사용량(출력 토큰 == 로컬 output_tokens)", u.get("output") == ok_call.get("output_tokens"), f"{u.get('output')}")
                gtxt = json.dumps(ok_gen["input"], ensure_ascii=False) if ok_gen else ""
                if cc is not None and isinstance(cc["output"], dict):
                    texts = {c["chunk_id"]: c["text"] for c in (record.get("final_context") or {}).get("chunks", [])}
                    absent = [cid for cid, t in texts.items() if json.dumps(t, ensure_ascii=False).strip('"')[:40] not in gtxt]
                    add("최종 문맥 본문이 모델 입력에 들어 있음", not absent, f"입력에 없는 청크 {absent}" if absent else f"{len(texts)}개 모두")
            else:
                add("실패한 모델 호출이 ERROR로 표시", all(g["level"] == "ERROR" for g in gens), f"{[g['level'] for g in gens]}")
    if err is None:
        cs = next((o for o in obs if o["name"] == "citation-and-status"), None)
        add("출처·상태 Span(citation-and-status)", cs is not None, "")
        if cs is not None and isinstance(cs["output"], dict):
            add("출처 변환 결과가 로컬 응답과 일치", cs["output"].get("citations") == record["response"]["citations"]
                and cs["output"].get("status") == record["response"]["status"], f"상태 {cs['output'].get('status')}")
    else:
        add("루트가 ERROR로 표시", root["level"] == "ERROR", str(root["level"]))
        e2 = root_out.get("error") or {}
        add("오류 유형 기록", e2.get("reason") == err["reason"], f"{e2.get('reason')}")
        add("오류 발생 단계 기록", e2.get("error_stage") == stage and f"stage={stage}" in str(root["status_message"]),
            f"단계 {e2.get('error_stage')} / {root['status_message']}")
        add("재시도 정보 기록", e2.get("retry_count") == err.get("retry_count") and "is_retryable" in e2,
            f"재시도 {e2.get('retry_count')}회")
    perf = root_out.get("performance") or {}
    add("실행시간 기록(total_duration_ms)", perf.get("total_duration_ms") == (record.get("performance") or {}).get("total_duration_ms"),
        f"{perf.get('total_duration_ms')}ms")
    n_secret = find_secrets(obs, env)
    add("비밀 값(OpenAI·Langfuse 비밀·Qdrant 키)이 기록에 없음", n_secret == 0, f"발견 {n_secret}건")
    pub = public_key_locations(obs, env)
    outside = {p: n for p, n in pub.items() if p != SDK_PUBLIC_KEY_PATH}
    add("Langfuse 공개 키는 SDK가 넣는 위치에만 있음(정보)", not outside,
        f"SDK metadata {pub.get(SDK_PUBLIC_KEY_PATH, 0)}곳" + (f", 그 밖의 위치 {sorted(outside)}" if outside else ""))
    return out


def check_batch(records: list[dict], observations_by_trace: dict[str, list[Any]], env: dict | None = None) -> dict:
    """요청 여러 건을 검사하고 요청 사이의 식별자 오류(Trace ID 중복, 누락)도 본다."""
    results = {}
    trace_ids = [r.get("trace_id") for r in records]
    for r in records:
        obs = observations_by_trace.get(r.get("trace_id") or "", [])
        checks = check_trace(r, obs, env) if obs else [("Langfuse에서 Trace를 찾음", False, "관측 0개")]
        results[r["request_id"]] = {"rag_method": r.get("rag_method"), "http_status": r.get("http_status"), "checks": checks}
    global_checks = [("요청마다 서로 다른 Trace ID", len(set(trace_ids)) == len(trace_ids) and all(trace_ids),
                      f"요청 {len(records)}건, 고유 Trace {len(set(trace_ids))}개"),
                     ("기록 누락 없음(모든 요청의 Trace가 있음)", all(observations_by_trace.get(t or "") for t in trace_ids), "")]
    return {"per_request": results, "global": global_checks,
            "passed": all(ok for v in results.values() for _, ok, _ in v["checks"]) and all(ok for _, ok, _ in global_checks)}
