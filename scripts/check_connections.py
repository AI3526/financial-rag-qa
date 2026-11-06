"""STEP 0 연결 확인 스크립트.

- .env 변수 설정 여부 확인 (값은 출력하지 않는다)
- OpenAI 모델 호출과 지원 파라미터 확인
- Langfuse 인증 확인, 테스트 Trace 1건 기록 후 도착 확인

사용: uv run python scripts/check_connections.py [env|openai|langfuse|all]
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from importlib.metadata import version
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

REQUIRED_SECRETS = ["OPENAI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
REQUIRED_PLAIN = ["LANGFUSE_BASE_URL", "OPENAI_MODEL"]
OPTIONAL = ["QDRANT_PATH", "HF_HOME"]

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{6,}"),
    re.compile(r"pk-lf-[A-Za-z0-9_\-]+"),
    re.compile(r"sk-lf-[A-Za-z0-9_\-]+"),
]


def redact(text: str) -> str:
    """오류 메시지 등에 키가 섞여 있어도 출력되지 않게 마스킹한다."""
    for env_name in REQUIRED_SECRETS:
        value = os.environ.get(env_name)
        if value:
            text = text.replace(value, "***")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("***", text)
    return text


def check_env() -> bool:
    ok = True
    print("[env] 변수 설정 여부 (값은 출력하지 않음)")
    for name in REQUIRED_SECRETS:
        is_set = bool(os.environ.get(name))
        ok &= is_set
        print(f"  {name}: {'set' if is_set else 'MISSING'}")
    for name in REQUIRED_PLAIN:
        value = os.environ.get(name)
        ok &= bool(value)
        shown = value if value else "MISSING"
        print(f"  {name}: {shown}")
    for name in OPTIONAL:
        print(f"  {name} (선택): {'set' if os.environ.get(name) else 'unset'}")
    return ok


def _usage_dict(usage) -> dict:
    if usage is None:
        return {}
    data = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
    keep = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        keep[key] = data.get(key)
    details = data.get("completion_tokens_details") or {}
    if details.get("reasoning_tokens") is not None:
        keep["reasoning_tokens"] = details.get("reasoning_tokens")
    return keep


def check_openai() -> dict | None:
    from openai import OpenAI

    model = os.environ["OPENAI_MODEL"]
    client = OpenAI()  # OPENAI_API_KEY는 환경변수에서 읽는다
    prompt = [{"role": "user", "content": "한 단어로만 답하세요: 안녕"}]
    print(f"[openai] model={model}, openai sdk={version('openai')}")

    variants = [
        ("기본 호출", {}),
        ("temperature=0", {"temperature": 0}),
        ("max_completion_tokens=200", {"max_completion_tokens": 200}),
        ("max_tokens=200", {"max_tokens": 200}),
    ]
    first_ok = None
    for label, extra in variants:
        started = time.perf_counter()
        try:
            resp = client.chat.completions.create(model=model, messages=prompt, **extra)
            elapsed = (time.perf_counter() - started) * 1000
            text = (resp.choices[0].message.content or "").strip()
            usage = _usage_dict(resp.usage)
            print(f"  OK   {label}: {elapsed:.0f}ms, returned_model={resp.model}, "
                  f"usage={usage}, text={text[:40]!r}")
            if first_ok is None:
                first_ok = {"text": text, "usage": usage, "model": resp.model, "prompt": prompt}
        except Exception as exc:  # noqa: BLE001 - 진단 목적
            status = getattr(exc, "status_code", None)
            code = getattr(exc, "code", None)
            msg = redact(str(exc))[:240].replace("\n", " ")
            print(f"  FAIL {label}: {type(exc).__name__} status={status} code={code} msg={msg}")
    return first_ok


def check_langfuse(openai_result: dict | None) -> bool:
    from langfuse import get_client, propagate_attributes

    print(f"[langfuse] sdk={version('langfuse')}, base_url={os.environ.get('LANGFUSE_BASE_URL')}")
    langfuse = get_client()
    try:
        authed = langfuse.auth_check()
    except Exception as exc:  # noqa: BLE001
        print(f"  auth_check 오류: {type(exc).__name__} {redact(str(exc))[:200]}")
        return False
    print(f"  auth_check: {authed}")
    if not authed:
        return False

    request_id = str(uuid.uuid4())
    trace_id = langfuse.create_trace_id(seed=request_id)
    print(f"  request_id={request_id}")
    print(f"  trace_id(결정적, seed=request_id)={trace_id}")
    again = langfuse.create_trace_id(seed=request_id)
    print(f"  같은 seed로 다시 생성한 값과 동일: {again == trace_id}")

    result = openai_result or {"text": "(OpenAI 결과 없음)", "usage": {}, "model": "n/a", "prompt": []}
    usage = result["usage"]
    with propagate_attributes(
        trace_name="connection-test",
        tags=["connection_test", "development"],
        metadata={"request_id": request_id, "track": "development", "step": "0"},
    ):
        with langfuse.start_as_current_observation(
            trace_context={"trace_id": trace_id},
            name="rag-query",
            as_type="span",
            input={"note": "STEP 0 연결 테스트"},
        ) as root:
            with langfuse.start_as_current_observation(
                name="answer-generation",
                as_type="generation",
                model=result["model"],
                input=result["prompt"],
            ) as gen:
                gen.update(
                    output=result["text"],
                    usage_details={
                        "input": usage.get("prompt_tokens") or 0,
                        "output": usage.get("completion_tokens") or 0,
                    }
                    if usage
                    else None,
                )
            root.update(output={"status": "connection_test_done"})
    langfuse.flush()

    url = langfuse.get_trace_url(trace_id=trace_id)
    print(f"  trace url: {url}")

    # 도착 확인 (실시간 조회 경로)
    for attempt in range(1, 7):
        time.sleep(3)
        try:
            resp = langfuse.api.observations.get_many(trace_id=trace_id, limit=10)
            names = [getattr(o, "name", None) for o in (getattr(resp, "data", None) or [])]
            print(f"  도착 확인 시도 {attempt}: observations={names}")
            if names:
                return True
        except Exception as exc:  # noqa: BLE001
            print(f"  도착 확인 시도 {attempt} 오류: {type(exc).__name__} {redact(str(exc))[:160]}")
    return False


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target in ("env", "all") and not check_env():
        print("필수 환경변수가 비어 있어 중단합니다.")
        return 1
    openai_result = None
    if target in ("openai", "all"):
        openai_result = check_openai()
    if target in ("langfuse", "all"):
        if not check_langfuse(openai_result):
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
