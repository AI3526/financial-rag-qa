"""확장 브랜치 2단계: 되묻기·보류일 때 답변이 밝힌 '모자란 점'(reason)으로 한 번 더 검색해 다시 답한다 (EXTENSION_INTERMEDIATE_LOG).

`RagPipeline`을 바꾸지 않고 `_generate`만 감싼 하위 클래스다(필수 구현·Basic 동작은 그대로).
- 첫 답변의 status가 `answered`면 아무것도 하지 않는다(질문당 호출 1회 그대로).
- `needs_clarification` 또는 `abstained`이면: 두 번째 질의 = `질문 + 첫 답변의 reason`(모델 출력만 사용, 평가 정답·근거 없음)으로 같은 검색기를 한 번 더 실행하고,
  첫 문맥의 상위 `keep_k`개 + 두 번째 검색에서 새로 나온 청크 `extra_k`개를 합쳐(같은 문맥 한도 4,000토큰 적용) 다시 생성한다. 두 번째 답변을 최종 응답으로 쓴다.
- run 레코드: `retry`(실행 여부, 첫 상태, 두 번째 질의, 더한 청크, 걸린 시간)와 첫 문맥(`first_final_context`)을 남기고 `final_context`는 두 번째 문맥이다.
- 모델 호출 상한(`max_model_calls`)은 두 번째 생성까지 합해 적용된다(재시도 포함).
"""
from __future__ import annotations

import time

from src.api.pipeline import RagPipeline
from src.generation.context import build_final_context

RETRY_STATUSES = ("needs_clarification", "abstained")
# broad 변형: answered여도 답변·이유에 '자료에서 확인하지 못했다'는 표현이 있으면 재검색한다(IF-DEV07처럼 일부만 답한 경우). 표현은 첫 답변에서 보인 문구로 정했다(과적합 가능성이 있다).
MISSING_MARKERS = ("확인되지 않", "확인할 수 없", "확인하지 못", "제시되지 않", "제공되지 않", "나와 있지 않", "찾을 수 없")


class RetryRagPipeline(RagPipeline):
    def __init__(self, *args, keep_k: int = 8, extra_k: int = 4, variant: str = "base", **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_k, self.extra_k = keep_k, extra_k
        if variant not in ("base", "broad", "reason"):
            raise ValueError(f"알 수 없는 variant: {variant}")
        self.variant = variant  # base: 되묻기·보류만, 질의=질문+reason / broad: answered라도 '확인되지 않음' 표현이 있으면 추가 / reason: 질의=reason만

    def _generate(self, question: str, sources: list[dict], record: dict) -> dict:
        parsed = super()._generate(question, sources, record)
        record["retry"] = {"triggered": False, "first_status": parsed["status"]}
        incomplete = self.variant == "broad" and parsed["status"] == "answered" and any(m in f"{parsed['answer']} {parsed['reason']}" for m in MISSING_MARKERS)
        if parsed["status"] not in RETRY_STATUSES and not incomplete:
            return parsed
        started = time.perf_counter()
        method = (record.get("retrieval") or {}).get("method") or "hybrid_rerank"
        query2 = parsed["reason"].strip() if self.variant == "reason" else f"{question} {parsed['reason']}".strip()
        result = self.retrievers[method].retrieve(query2, k=self.keep_k + self.extra_k)
        have = {s["chunk_id"] for s in sources[: self.keep_k]}
        extras = [c for c in result.chunks if c["chunk_id"] not in have][: self.extra_k]
        merged = sources[: self.keep_k] + extras
        context = build_final_context(merged, self.settings.context_limit_tokens)
        record["retry"] = {"triggered": True, "first_status": parsed["status"], "variant": self.variant, "query": query2, "added_chunks": [
            {"chunk_id": c["chunk_id"], "book_id": c["book_id"], "page": c["page"]} for c in extras],
            "first_final_context": record["final_context"], "retrieval_duration_ms": round((time.perf_counter() - started) * 1000, 1)}
        record["final_context"] = context.as_record()
        return super()._generate(question, context.sources, record)
