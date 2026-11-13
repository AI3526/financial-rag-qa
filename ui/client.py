"""[확장] UI가 API 서버(`POST /rag/query`)를 부르는 부분. 화면(Streamlit) 코드와 분리해 네트워크 없이 테스트한다.

- 서버 계약은 바꾸지 않는다: 요청 본문은 `{question, rag_method}` 두 필드뿐이다(평가 정보 `question_id` 등은 보내지 않는다).
- UI 요청은 `X-Track: extension`으로 표시해 공식 평가(official)와 섞이지 않게 한다. `X-Experiment-Id`는 보내지 않으므로
  서버가 `results/service/runs.jsonl`(Git 제외)에 기록한다.
- 이 단계에는 대화 메모리와 캐시가 없다. 질문마다 독립 요청이다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 120.0  # 서버의 모델 호출(최대 3회 시도, 호출당 timeout 30초)보다 길게 기다린다

# 화면에 보이는 이름. 첫 번째가 기본값이다(공식 선택은 hybrid_rerank).
RAG_METHODS = {
    "hybrid_rerank": "Hybrid + Rerank (선택된 방식)",
    "hybrid": "Hybrid",
    "dense": "Dense",
}

STATUS_LABELS = {
    "answered": "답변",
    "needs_clarification": "추가 확인 필요",
    "abstained": "근거 부족",
    "error": "오류",
}

REQUEST_HEADERS = {"X-Track": "extension", "X-Run-Type": "general_eval"}

# 서버의 reason 코드(09 §8.5)와 UI가 직접 만드는 코드를 사람이 읽는 문장으로 바꾼다.
ERROR_MESSAGES = {
    "invalid_request": "질문 형식이 올바르지 않습니다. 질문을 입력해 주세요.",
    "retrieval_error": "문서를 검색하는 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
    "model_error": "답변을 만드는 모델 호출에 실패했습니다. 잠시 후 다시 시도해 주세요.",
    "model_timeout": "답변을 만드는 데 시간이 너무 오래 걸렸습니다. 잠시 후 다시 시도해 주세요.",
    "output_validation_error": "모델 응답이 형식에 맞지 않아 답변을 만들지 못했습니다. 질문을 바꿔 다시 시도해 주세요.",
    "internal_error": "서버 내부 오류가 발생했습니다. 서버 설정(OPENAI_API_KEY 등)을 확인해 주세요.",
    # UI가 서버를 부르기 전에 막거나 서버에 닿지 못한 경우
    "empty_question": "질문을 입력해 주세요.",
    "invalid_method": "지원하지 않는 검색 방식입니다.",
    "server_unreachable": "서버에 연결할 수 없습니다. API 서버가 켜져 있는지, 주소가 맞는지 확인해 주세요.",
    "ui_timeout": "서버 응답을 기다리다 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.",
    "bad_response": "서버 응답을 읽을 수 없습니다. 서버 버전과 주소를 확인해 주세요.",
}


@dataclass(frozen=True)
class SourceRef:
    """같은 (문서, 쪽)을 한 줄로 묶은 출처. `source_ids`는 답변이 인용한 S1, S2 같은 번호다."""
    book_id: str
    file_name: str
    page: int
    source_ids: tuple[str, ...]


@dataclass
class UiAnswer:
    status: str
    answer: str
    reason: str
    rag_method: str
    sources: list[SourceRef] = field(default_factory=list)
    request_id: str | None = None
    trace_id: str | None = None
    http_status: int | None = None
    elapsed_ms: int | None = None
    error_code: str | None = None  # 서버가 준 reason(오류일 때) 또는 UI 쪽 오류 코드

    @property
    def is_error(self) -> bool:
        return self.status == "error"

    @property
    def status_label(self) -> str:
        return STATUS_LABELS.get(self.status, self.status)


def describe_error(code: str | None) -> str:
    return ERROR_MESSAGES.get(code or "", "알 수 없는 오류가 발생했습니다.")


def group_sources(citations: list[dict]) -> list[SourceRef]:
    """citations를 (book_id, file_name, page) 단위로 묶고 처음 나온 순서를 유지한다."""
    grouped: dict[tuple[str, str, int], list[str]] = {}
    for c in citations:
        key = (str(c.get("book_id", "")), str(c.get("file_name", "")), int(c.get("page", 0)))
        ids = grouped.setdefault(key, [])
        sid = str(c.get("source_id", ""))
        if sid and sid not in ids:
            ids.append(sid)
    return [SourceRef(book_id=b, file_name=f, page=p, source_ids=tuple(ids)) for (b, f, p), ids in grouped.items()]


def _local_error(code: str, rag_method: str, elapsed_ms: int | None = None) -> UiAnswer:
    return UiAnswer(status="error", answer=describe_error(code), reason=code, rag_method=rag_method,
                    elapsed_ms=elapsed_ms, error_code=code)


def parse_response(resp: httpx.Response, rag_method: str, elapsed_ms: int) -> UiAnswer:
    """서버 응답을 화면용 값으로 바꾼다. 형식이 다르면 `bad_response` 오류로 돌려준다."""
    try:
        body = resp.json()
    except ValueError:
        return _local_error("bad_response", rag_method, elapsed_ms)
    if not isinstance(body, dict) or body.get("status") not in STATUS_LABELS:
        return _local_error("bad_response", rag_method, elapsed_ms)
    status = body["status"]
    answer = body.get("answer") if isinstance(body.get("answer"), str) else ""
    reason = body.get("reason") if isinstance(body.get("reason"), str) else ""
    citations = body.get("citations") if isinstance(body.get("citations"), list) else []
    try:
        sources = group_sources([c for c in citations if isinstance(c, dict)])
    except (TypeError, ValueError):
        return _local_error("bad_response", rag_method, elapsed_ms)
    code = reason if status == "error" else None
    if status == "error":  # 서버의 일반 문구 대신 원인별 안내를 보여 준다
        answer = describe_error(code)
    return UiAnswer(status=status, answer=answer, reason=reason, rag_method=rag_method, sources=sources,
                    request_id=body.get("request_id") or resp.headers.get("X-Request-Id"),
                    trace_id=resp.headers.get("X-Trace-Id"), http_status=resp.status_code,
                    elapsed_ms=elapsed_ms, error_code=code)


def ask(question: str, rag_method: str, *, base_url: str = DEFAULT_BASE_URL, timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.Client | None = None) -> UiAnswer:
    """질문 하나를 서버에 보내고 결과를 돌려준다. 빈 질문과 지원하지 않는 방식은 서버를 부르지 않는다."""
    q = question.strip() if isinstance(question, str) else ""
    if not q:
        return _local_error("empty_question", rag_method)
    if rag_method not in RAG_METHODS:
        return _local_error("invalid_method", rag_method)
    owns_client = client is None
    http = client or httpx.Client(timeout=timeout)
    start = time.perf_counter()
    try:
        resp = http.post(f"{base_url.rstrip('/')}/rag/query", json={"question": q, "rag_method": rag_method}, headers=REQUEST_HEADERS)
    except httpx.TimeoutException:
        return _local_error("ui_timeout", rag_method, int((time.perf_counter() - start) * 1000))
    except httpx.TransportError:
        return _local_error("server_unreachable", rag_method, int((time.perf_counter() - start) * 1000))
    finally:
        if owns_client:
            http.close()
    return parse_response(resp, rag_method, int((time.perf_counter() - start) * 1000))


def check_health(*, base_url: str = DEFAULT_BASE_URL, timeout: float = 5.0, client: httpx.Client | None = None) -> bool:
    """`GET /health`가 응답하는지만 확인한다."""
    owns_client = client is None
    http = client or httpx.Client(timeout=timeout)
    try:
        resp = http.get(f"{base_url.rstrip('/')}/health")
        return resp.status_code == 200 and resp.json().get("status") == "ok"
    except (httpx.HTTPError, ValueError, AttributeError):
        return False
    finally:
        if owns_client:
            http.close()
