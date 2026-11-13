"""[확장] UI 클라이언트 테스트: httpx.MockTransport로 서버를 흉내 낸다 (네트워크·모델 불필요)."""
import json

import httpx
import pytest

from ui import client as ui


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _ok_body(**over):
    body = {"request_id": "req-1", "status": "answered", "answer": "답변입니다.", "reason": "근거 확인",
            "citations": [{"source_id": "S1", "book_id": "FB01", "file_name": "FB01.pdf", "page": 2},
                          {"source_id": "S2", "book_id": "FB01", "file_name": "FB01.pdf", "page": 2},
                          {"source_id": "S3", "book_id": "FB05", "file_name": "FB05.pdf", "page": 15}]}
    body.update(over)
    return body


def test_정상_응답을_화면용_값으로_바꾼다():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["headers"], seen["body"] = str(request.url), request.headers, json.loads(request.content)
        return httpx.Response(200, json=_ok_body(), headers={"X-Request-Id": "req-1", "X-Trace-Id": "t" * 32})

    r = ui.ask("  루미센서 E는 어떤 제품인가요?  ", "hybrid_rerank", base_url="http://x:8000/", client=_client(handler))
    assert (r.status, r.answer, r.reason, r.http_status) == ("answered", "답변입니다.", "근거 확인", 200)
    assert (r.request_id, r.trace_id, r.error_code, r.is_error) == ("req-1", "t" * 32, None, False)
    assert r.status_label == "답변" and r.elapsed_ms is not None
    assert seen["url"] == "http://x:8000/rag/query"  # 끝의 / 정리


def test_출처는_같은_문서와_쪽을_한_줄로_묶고_순서를_지킨다():
    r = ui.ask("질문", "dense", client=_client(lambda req: httpx.Response(200, json=_ok_body())))
    assert [(s.file_name, s.page, s.source_ids) for s in r.sources] == [("FB01.pdf", 2, ("S1", "S2")), ("FB05.pdf", 15, ("S3",))]


def test_요청은_질문과_방식만_보내고_확장_헤더를_붙인다():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"], seen["headers"] = json.loads(request.content), request.headers
        return httpx.Response(200, json=_ok_body())

    ui.ask("질문", "hybrid", client=_client(handler))
    assert seen["body"] == {"question": "질문", "rag_method": "hybrid"}  # question_id 등 추가 필드 없음
    assert seen["headers"]["X-Track"] == "extension" and seen["headers"]["X-Run-Type"] == "general_eval"
    assert "x-experiment-id" not in seen["headers"]  # 공식 실험 ID를 쓰지 않는다


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_빈_질문은_서버를_부르지_않는다(question):
    calls = []
    r = ui.ask(question, "dense", client=_client(lambda req: calls.append(req) or httpx.Response(200, json=_ok_body())))
    assert calls == [] and r.is_error and r.error_code == "empty_question" and r.answer == ui.describe_error("empty_question")


def test_지원하지_않는_방식은_서버를_부르지_않는다():
    calls = []
    r = ui.ask("질문", "bm25", client=_client(lambda req: calls.append(req) or httpx.Response(200, json=_ok_body())))
    assert calls == [] and r.error_code == "invalid_method"


@pytest.mark.parametrize("http_status,reason", [(422, "invalid_request"), (500, "retrieval_error"), (500, "internal_error"),
                                                (502, "model_error"), (502, "output_validation_error"), (504, "model_timeout")])
def test_서버_오류는_원인별_안내로_바뀐다(http_status, reason):
    body = {"request_id": "req-9", "status": "error", "answer": "서버 일반 문구", "reason": reason, "citations": []}
    r = ui.ask("질문", "dense", client=_client(lambda req: httpx.Response(http_status, json=body, headers={"X-Request-Id": "req-9"})))
    assert r.is_error and r.error_code == reason and r.http_status == http_status
    assert r.answer == ui.describe_error(reason) != "서버 일반 문구"
    assert r.request_id == "req-9" and r.sources == []


@pytest.mark.parametrize("status", ["needs_clarification", "abstained"])
def test_되묻기와_보류는_오류가_아니다(status):
    r = ui.ask("질문", "dense", client=_client(lambda req: httpx.Response(200, json=_ok_body(status=status, answer="확인해 주세요.", citations=[]))))
    assert not r.is_error and r.status == status and r.error_code is None and r.sources == [] and r.answer == "확인해 주세요."


def test_서버에_연결할_수_없으면_안내한다():
    def handler(request):
        raise httpx.ConnectError("refused")

    r = ui.ask("질문", "dense", client=_client(handler))
    assert r.is_error and r.error_code == "server_unreachable" and r.http_status is None


def test_응답이_늦으면_시간_초과_안내를_한다():
    def handler(request):
        raise httpx.ReadTimeout("slow")

    r = ui.ask("질문", "dense", client=_client(handler))
    assert r.is_error and r.error_code == "ui_timeout"


@pytest.mark.parametrize("response", [httpx.Response(200, text="<html>not json</html>"),
                                      httpx.Response(200, json=["list"]),
                                      httpx.Response(200, json={"status": "unknown", "answer": "a"}),
                                      httpx.Response(200, json={"status": "answered", "citations": [{"page": "x"}]})])
def test_형식이_다른_응답은_bad_response로_처리한다(response):
    r = ui.ask("질문", "dense", client=_client(lambda req: response))
    assert r.is_error and r.error_code == "bad_response"


def test_상태_확인():
    assert ui.check_health(client=_client(lambda req: httpx.Response(200, json={"status": "ok"}))) is True
    assert ui.check_health(client=_client(lambda req: httpx.Response(500, json={}))) is False
    assert ui.check_health(client=_client(lambda req: httpx.Response(200, text="x"))) is False

    def refuse(request):
        raise httpx.ConnectError("refused")

    assert ui.check_health(client=_client(refuse)) is False


def test_기본_검색_방식은_선택된_hybrid_rerank다():
    assert next(iter(ui.RAG_METHODS)) == "hybrid_rerank"
    assert set(ui.RAG_METHODS) == {"dense", "hybrid", "hybrid_rerank"}
