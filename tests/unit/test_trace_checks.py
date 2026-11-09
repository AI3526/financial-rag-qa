import copy
import json

from src.api import pipeline as pl
from src.api.app import _trace_output
from src.api.pipeline import PipelineSettings, RagPipeline
from src.observability import trace_checks as tc
from tests.unit.test_api import OK, FakeRetriever, ScriptedGenerator

TID = "a" * 32


def run_pipeline(generator=None, retriever=None, method="dense"):
    pipe = RagPipeline(retriever or FakeRetriever(), generator or ScriptedGenerator(OK), PipelineSettings(), sleep=lambda s: None)
    out = pipe.run("req-1234-abcd", "영업이익은?", method)
    rec = {**out.record, "request_id": "req-1234-abcd", "trace_id": TID, "run_type": "general_eval", "track": "development",
           "rag_method": method, "response": out.response, "http_status": out.http_status,
           "versions": {"config_version": "cfg-0123456789"}, "observability": {"langfuse_recorded": True, "langfuse_error": None}}
    return rec, out


def obs(id, name, parent="r", type="SPAN", **kw):
    return {"id": id, "name": name, "type": type, "parent_observation_id": parent, "trace_id": TID, "level": "DEFAULT",
            "status_message": None, "input": None, "output": None, "metadata": {}, "model": None, "usage_details": {}, **kw}


MD = {"request_id": "req-1234-abcd", "run_type": "general_eval", "track": "development", "rag_method": "dense",
      "config_version": "cfg-0123456789", "dataset_id": "basic", "dataset_version": "basic-v1", "prompt_version": "answer-v2"}


def context_obs(rec):
    ctx = rec["final_context"]["chunks"]
    return obs("c", "context-construction", input={"retrieved": [{"chunk_id": c["chunk_id"]} for c in ctx]},
               output={"final_chunks": [{"chunk_id": c["chunk_id"]} for c in ctx], "dropped_chunk_ids": []})


def good_success():
    rec, out = run_pipeline()
    ctx = rec["final_context"]["chunks"]
    prompt = [{"role": "user", "content": "\n".join(f"[{c['source_id']}] {c['text']}" for c in ctx)}]
    call = rec["llm_calls"][0]
    observations = [
        obs("r", "rag-query", parent=None, metadata=dict(MD), output=_trace_output(out)),
        obs("d", "dense-retrieve", type="RETRIEVER"),
        context_obs(rec),
        obs("g", "answer-generation", type="GENERATION", input=prompt, output={"role": "assistant", "content": "{}"},
            model="gpt-6-luna",
            usage_details={"input": call["input_tokens"] - 50, "input_cache_read": 50, "output": call["output_tokens"]}),
        obs("s", "citation-and-status", output={"validation": "passed", "status": "answered", "citations": rec["response"]["citations"]}),
    ]
    return rec, observations


def failed(checks):
    return [n for n, ok, _ in checks if not ok]


def test_a_correct_trace_passes_every_check():
    rec, observations = good_success()
    checks = tc.check_trace(rec, observations, env={})
    assert failed(checks) == [] and len(checks) >= 14


def test_missing_context_construction_is_detected():
    rec, o = good_success()
    o = [x for x in o if x["name"] != "context-construction"]
    assert "최종 문맥 Span(context-construction)" in failed(tc.check_trace(rec, o, env={}))


def test_request_id_or_trace_id_mismatch_is_detected():
    rec, o = good_success()
    bad = copy.deepcopy(o)
    bad[0]["metadata"]["request_id"] = "other-request"
    assert "Request ID 연결(metadata.request_id)" in failed(tc.check_trace(rec, bad, env={}))
    wrong_trace = copy.deepcopy(o)
    wrong_trace[1]["trace_id"] = "b" * 32
    assert "Trace ID 연결(로컬 trace_id == Langfuse)" in failed(tc.check_trace(rec, wrong_trace, env={}))


def test_missing_required_metadata_is_detected():
    rec, o = good_success()
    o[0]["metadata"].pop("config_version")
    o[0]["metadata"].pop("dataset_id")
    f = failed(tc.check_trace(rec, o, env={}))
    assert "필수 식별 정보(metadata)" in f and "config_version 일치" in f


def test_orphan_observation_and_duplicate_roots_are_detected():
    rec, o = good_success()
    orphan = copy.deepcopy(o)
    orphan.append(obs("x", "dense-retrieve", parent="missing", type="RETRIEVER"))
    assert "모든 관측이 루트 아래에 연결" in failed(tc.check_trace(rec, orphan, env={}))
    two = copy.deepcopy(o) + [obs("r2", "rag-query", parent=None, metadata={"request_id": "req-1234-abcd"})]
    assert "루트 Trace 1개(rag-query)" in failed(tc.check_trace(rec, two, env={}))


def test_final_context_that_differs_from_the_local_record_or_model_input_is_detected():
    rec, o = good_success()
    bad = copy.deepcopy(o)
    bad[2]["output"]["final_chunks"].pop()
    f = failed(tc.check_trace(rec, bad, env={}))
    assert "최종 문맥 청크가 로컬 기록과 일치" in f and "검색 결과와 최종 문맥의 구분(검색 − 제외 = 최종)" in f
    bad2 = copy.deepcopy(o)
    bad2[3]["input"] = [{"role": "user", "content": "전혀 다른 프롬프트"}]
    assert "최종 문맥 본문이 모델 입력에 들어 있음" in failed(tc.check_trace(rec, bad2, env={}))


def test_generation_count_must_match_local_calls_including_retries():
    rec, o = good_success()
    two = copy.deepcopy(o) + [obs("g2", "answer-generation", type="GENERATION", input=[{"x": 1}], model="gpt-6-luna")]
    assert "모델 호출(answer-generation) 수 == 로컬 llm_calls 수(재시도 포함)" in failed(tc.check_trace(rec, two, env={}))
    renamed = copy.deepcopy(o)
    renamed[3]["name"] = "ChatOpenAI"
    assert "모델 호출(answer-generation) 수 == 로컬 llm_calls 수(재시도 포함)" in failed(tc.check_trace(rec, renamed, env={}))


def test_usage_mismatch_is_detected():
    rec, o = good_success()
    o[3]["usage_details"]["output"] += 5
    assert "사용량(출력 토큰 == 로컬 output_tokens)" in failed(tc.check_trace(rec, o, env={}))


def test_citation_mismatch_is_detected():
    rec, o = good_success()
    o[4]["output"]["citations"] = []
    assert "출처 변환 결과가 로컬 응답과 일치" in failed(tc.check_trace(rec, o, env={}))


def test_secrets_in_the_record_are_counted_without_exposing_values():
    rec, o = good_success()
    env = {"OPENAI_API_KEY": "sk-proj-SECRETVALUE0123456789"}
    assert tc.find_secrets([tc.normalize(x) for x in o], env=env) == 0
    o[3]["input"] = [{"role": "user", "content": "키는 sk-proj-SECRETVALUE0123456789 입니다"}]
    checks = tc.check_trace(rec, o, env=env)
    assert "비밀 값(OpenAI·Langfuse 비밀·Qdrant 키)이 기록에 없음" in failed(checks)
    assert "SECRETVALUE" not in json.dumps(checks, ensure_ascii=False)  # 결과에도 값이 없다


def test_key_shaped_strings_are_detected_even_without_the_env_value():
    rec, o = good_success()
    o[1]["output"] = {"leak": "sk-lf-12345678-aaaa"}
    assert tc.find_secrets([tc.normalize(x) for x in o], env={}) == 1


def test_langfuse_public_key_is_allowed_only_where_the_sdk_puts_it():
    rec, o = good_success()
    env = {"LANGFUSE_PUBLIC_KEY": "pk-lf-abcdef12-3456"}
    for x in o:
        x["metadata"] = {**x["metadata"], "scope": {"attributes": {"public_key": "pk-lf-abcdef12-3456"}}}
    checks = tc.check_trace(rec, o, env=env)
    assert failed(checks) == []  # SDK가 넣은 위치의 공개 키는 비밀 값 실패가 아니다
    assert dict(tc.public_key_locations([tc.normalize(x) for x in o], env)) == {tc.SDK_PUBLIC_KEY_PATH: len(o)}
    o[3]["input"] = [{"role": "user", "content": "공개 키 pk-lf-abcdef12-3456 를 프롬프트에 넣음"}]
    assert "Langfuse 공개 키는 SDK가 넣는 위치에만 있음(정보)" in failed(tc.check_trace(rec, o, env=env))
    assert "pk-lf-abcdef12" not in json.dumps(tc.check_trace(rec, o, env=env), ensure_ascii=False)  # 결과에 값이 없다
    # 공개 키 이름의 환경 변수는 비밀 목록에 없다
    assert "LANGFUSE_PUBLIC_KEY" not in tc.SECRET_ENV_NAMES and "LANGFUSE_SECRET_KEY" in tc.SECRET_ENV_NAMES


def error_case(generator):
    rec, out = run_pipeline(generator=generator)
    err = rec["error"]
    gens = [obs(f"g{i}", "answer-generation", type="GENERATION", input=[{"x": 1}], model="gpt-6-luna", level="ERROR")
            for i, _ in enumerate(rec["llm_calls"])]
    root = obs("r", "rag-query", parent=None, metadata=dict(MD), level="ERROR",
               status_message=f"{err['reason']} (stage={err['error_stage']})", output=_trace_output(out))
    return rec, [root, obs("d", "dense-retrieve", type="RETRIEVER"), context_obs(rec), *gens]


def test_error_trace_with_retries_passes_and_a_missing_stage_fails():
    rec, o = error_case(ScriptedGenerator(pl.ModelTimeout("x")))
    assert len(rec["llm_calls"]) == 3 and rec["error"]["error_stage"] == "generation"
    assert failed(tc.check_trace(rec, o, env={})) == []
    bad = copy.deepcopy(o)
    bad[0]["output"]["error"]["error_stage"] = "retrieval"
    bad[0]["status_message"] = "model_timeout"
    assert "오류 발생 단계 기록" in failed(tc.check_trace(rec, bad, env={}))
    bad2 = copy.deepcopy(o)
    bad2[0]["level"] = "DEFAULT"
    assert "루트가 ERROR로 표시" in failed(tc.check_trace(rec, bad2, env={}))
    bad3 = copy.deepcopy(o)[:-1]  # 재시도 3번 중 하나의 Generation이 Langfuse에 없다
    assert "모델 호출(answer-generation) 수 == 로컬 llm_calls 수(재시도 포함)" in failed(tc.check_trace(rec, bad3, env={}))


def test_batch_detects_missing_traces_and_duplicate_trace_ids():
    rec, o = good_success()
    ok = tc.check_batch([rec], {TID: o}, env={})
    assert ok["passed"] is True
    missing = tc.check_batch([rec], {}, env={})
    assert missing["passed"] is False and ("기록 누락 없음(모든 요청의 Trace가 있음)", False, "") in missing["global"]
    rec2 = {**rec, "request_id": "req-other"}
    dup = tc.check_batch([rec, rec2], {TID: o}, env={})
    assert dup["passed"] is False and dup["global"][0][1] is False  # 같은 Trace ID를 두 요청이 쓰면 오류
