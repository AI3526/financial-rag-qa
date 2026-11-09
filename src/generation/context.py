"""최종 문맥 구성 (STEP 8, 08 §12.2, 04 §4.6).

- 검색 결과(rank 순)에 이번 요청에서만 쓰는 출처 ID(S1, S2…)를 붙이고 문서·페이지 대응표를 코드가 보관한다.
- 토큰 한도(초기값 4,000)를 넘으면 **낮은 순위 청크를 통째로 제외**한다. 청크를 자르지 않으므로 대상·조건·단위·
  정정 내용·출처가 보존된다. 남은 청크의 출처 ID는 다시 S1부터 순서대로 매긴다.
- gpt-6-luna의 공식 토크나이저를 확인하지 못해 `tiktoken o200k_base`로 **추정**한다. 추정치이며 오차는 D-032에 기록한다.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field

TOKEN_COUNT_METHOD = "tiktoken-o200k_base (estimate)"
NO_CONTEXT_TEXT = "(검색된 자료 없음)"


@functools.lru_cache(maxsize=1)
def _encoder():
    import tiktoken

    return tiktoken.get_encoding("o200k_base")


def estimate_tokens(text: str) -> int:
    return len(_encoder().encode(text))


def build_sources(chunks: list[dict]) -> list[dict]:
    """청크에 출처 ID(S1, S2…)를 붙인다. 문서명·페이지는 이 대응표로만 변환하며 LLM이 만들지 않는다."""
    return [
        {"source_id": f"S{i}", "chunk_id": c["chunk_id"], "book_id": c["book_id"], "file_name": c["file_name"],
         "page": c["page"], "rank": c["rank"], "text": c["text"]}
        for i, c in enumerate(chunks, start=1)
    ]


def format_source(source: dict) -> str:
    return f"[{source['source_id']}] {source['text']}"


def format_context(sources: list[dict]) -> str:
    return "\n\n".join(format_source(s) for s in sources) if sources else NO_CONTEXT_TEXT


@dataclass
class FinalContext:
    sources: list[dict]
    limit_tokens: int
    token_count: int  # 추정
    dropped_chunk_ids: list[str] = field(default_factory=list)
    over_limit: bool = False  # 첫 청크 하나만으로 한도를 넘어 그대로 둔 경우

    def as_record(self) -> dict:
        return {"context_limit_tokens": self.limit_tokens, "context_token_count": self.token_count,
                "token_count_method": TOKEN_COUNT_METHOD, "dropped_chunk_ids": self.dropped_chunk_ids,
                "over_limit": self.over_limit, "chunks": self.sources}


def build_final_context(chunks: list[dict], limit_tokens: int) -> FinalContext:
    """rank 순으로 한도 안에 드는 만큼만 담는다. 다음 청크가 들어가지 않으면 그 뒤 청크도 모두 제외한다."""
    kept: list[dict] = []
    used, over = 0, False
    for chunk in chunks:
        cost = estimate_tokens(format_source({"source_id": f"S{len(kept) + 1}", "text": chunk["text"]})) + 2  # 구분 줄바꿈
        if used + cost > limit_tokens:
            if not kept:  # 1위 청크만으로 넘는다: 자르지 않고 그대로 두되 표시한다(자료 없음으로 오판하지 않기 위해)
                kept.append(chunk)
                used, over = cost, True
            break
        kept.append(chunk)
        used += cost
    dropped = [c["chunk_id"] for c in chunks[len(kept):]]
    return FinalContext(sources=build_sources(kept), limit_tokens=limit_tokens, token_count=used,
                        dropped_chunk_ids=dropped, over_limit=over)
