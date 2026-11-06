"""청킹 (STEP 2).

`pages.jsonl`의 페이지 텍스트를 검색 단위(청크)로 나눠 `chunks.jsonl` 행을 만든다.
필드 정의는 project_docs/02_implementation/09_DATA_SCHEMA.md §6.3, 설계 결정은 DECISIONS.md D-025를 따른다.

설계 (chunk-v1)
- 청크는 페이지 경계를 넘지 않고 단일 `page`를 가진다 (D-014).
- 페이지 구조: 머리말 1줄 + 문서 제목 1줄 + 소제목 1줄 + 본문 + 꼬리말(`N / M`). 머리말·제목·소제목은 청크 본문에서 빼고
  `metadata`로 보존하며, 검색용 텍스트(`retrieval_text`)의 앞에 붙인다. 꼬리말은 버린다 (쪽 번호는 `page`로 보존).
- 문단 나누기: PDF가 줄 폭을 고정해 줄을 나누므로, 줄 길이가 그 페이지 최장 줄의 85% 미만이면 문단의 끝으로 본다.
- 줄 잇기: 빈 문자로 잇는다. 줄이 단어 중간에서 끊기므로 공백을 넣으면 고유명사(해온솔루션 등)가 쪼개진다.
  다만 줄이 문장부호(. ? ! , ; : ) 등)로 끝나면 단어 경계가 확실하므로 공백을 넣고, 다음 줄이 공백으로 시작하면
  (원본 PDF에 실제로 있는 공백이므로) 그대로 보존한다.
- 묶기: 같은 페이지의 연속 문단을 `max_chars`까지 묶는다. 문단 하나가 `max_chars`를 넘어도 `hard_max_chars`까지는
  쪼개지 않고 통째로 하나의 청크로 둔다 (문단을 문장 중간에서 가르면 근거가 청크 경계에 걸린다). `hard_max_chars`를 넘는
  문단만 문장 단위로 나눈다.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

CHUNKING_METHOD = "paragraph-pack"
CHUNKING_VERSION = "chunk-v1"

# 줄이 이 문자로 끝나면 다음 줄과의 경계는 단어 경계이므로 공백을 넣는다
_BOUNDARY_END = ".?!,;:)”’"
_FOOTER_RE = re.compile(r"\d+\s*/\s*\d+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!])\s+")


@dataclass(frozen=True)
class ChunkParams:
    max_chars: int = 450            # 여러 문단을 묶을 때의 한도
    hard_max_chars: int = 800       # 문단 하나를 통째로 둘 수 있는 한도. 넘으면 문장 단위로 나눈다 (e5 입력 한도 512토큰 이내)
    overlap_units: int = 0          # 다음 청크에 되풀이할 문단(또는 문장) 수. 0이면 overlap 없음
    paragraph_ratio: float = 0.85   # 줄 길이가 (페이지 최장 줄 × ratio) 미만이면 문단의 끝

    def as_dict(self) -> dict:
        return {
            "max_chars": self.max_chars,
            "hard_max_chars": self.hard_max_chars,
            "overlap_units": self.overlap_units,
            "paragraph_ratio": self.paragraph_ratio,
            "join_rule": "empty-join; space after sentence punctuation; keep original leading space",
        }


def _nfkc_len(text: str) -> int:
    return len(unicodedata.normalize("NFKC", text))


# --- 페이지 구조 ---
def parse_page(text: str) -> tuple[dict, list[str], bool]:
    """페이지 텍스트를 (머리말·제목·소제목 metadata, 본문 줄 목록, 구조가 예상과 같은지)로 나눈다.

    구조가 예상과 다르면(줄 수가 모자라거나 꼬리말이 없음) 전체를 본문으로 보고 metadata는 비운다.
    """
    lines = text.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) >= 5 and _FOOTER_RE.fullmatch(lines[-1].strip()):
        meta = {
            "header": lines[0].strip(),
            "doc_title": lines[1].strip(),
            "section_title": lines[2].strip(),
        }
        return meta, lines[3:-1], True
    return {}, lines, False


def split_paragraph_lines(body_lines: list[str], ratio: float) -> list[list[str]]:
    """줄 길이로 문단을 나눈다. 빈 줄은 건너뛴다."""
    lines = [l for l in body_lines if l.strip()]
    if not lines:
        return []
    lengths = [_nfkc_len(l) for l in lines]
    threshold = ratio * max(lengths)
    paragraphs: list[list[str]] = []
    current: list[str] = []
    for line, length in zip(lines, lengths):
        current.append(line)
        if length < threshold:
            paragraphs.append(current)
            current = []
    if current:
        paragraphs.append(current)
    return paragraphs


def join_lines(lines: list[str]) -> str:
    """고정 폭으로 나뉜 줄들을 한 문단으로 잇는다 (모듈 설명의 줄 잇기 규칙)."""
    out = lines[0].strip()
    for nxt in lines[1:]:
        nxt = nxt.rstrip()
        if not nxt.strip():
            continue
        if out and out[-1] in _BOUNDARY_END and not nxt[0].isspace():
            out += " " + nxt
        else:
            out += nxt  # 다음 줄이 공백으로 시작하면 그 공백이 그대로 이어진다
    return out


# --- 묶기 ---
def _split_long(text: str, max_chars: int) -> list[str]:
    """max_chars를 넘는 문단을 문장 단위로 나누고, 그래도 긴 문장은 공백 위치에서 자른다."""
    pieces: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        sentence = sentence.strip()
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)
    return pieces


def _page_units(paragraphs: list[str], params: ChunkParams) -> list[tuple[str, int]]:
    """(텍스트, 문단 번호) 단위 목록. 문단은 hard_max_chars까지 통째로 두고, 넘는 문단만 문장 단위로 나눈다."""
    units: list[tuple[str, int]] = []
    for idx, para in enumerate(paragraphs):
        if len(para) <= params.hard_max_chars:
            units.append((para, idx))
        else:
            units.extend((piece, idx) for piece in _split_long(para, params.max_chars))
    return units


def _render(units: list[tuple[str, int]]) -> str:
    out = units[0][0]
    for (prev_text, prev_idx), (text, idx) in zip(units, units[1:]):
        out += (" " if idx == prev_idx else "\n") + text
    return out


def pack_units(units: list[tuple[str, int]], params: ChunkParams) -> list[list[tuple[str, int]]]:
    groups: list[list[tuple[str, int]]] = []
    current: list[tuple[str, int]] = []
    for unit in units:
        candidate = current + [unit]
        if current and len(_render(candidate)) > params.max_chars:
            groups.append(current)
            carried = current[-params.overlap_units:] if params.overlap_units > 0 else []
            # 되풀이한 단위와 새 단위가 한도를 넘으면 되풀이를 포기한다
            current = carried if carried and len(_render(carried + [unit])) <= params.max_chars else []
        current = current + [unit]
    if current:
        groups.append(current)
    return groups


# --- 페이지·문서 단위 ---
def chunk_page(page: dict, params: ChunkParams) -> tuple[list[dict], bool]:
    """페이지 1개를 청크 목록으로 만든다. 반환: (청크 필드 목록, 구조가 예상과 같은지).

    청크에는 chunk_text, page, metadata만 채우고 나머지 식별 필드는 chunk_document가 붙인다.
    """
    meta, body_lines, structured = parse_page(page["text"])
    paragraphs = [join_lines(p) for p in split_paragraph_lines(body_lines, params.paragraph_ratio)]
    units = _page_units(paragraphs, params)
    chunks = []
    for group in pack_units(units, params) if units else []:
        chunks.append({"page": page["page"], "chunk_text": _render(group), "metadata": dict(meta)})
    return chunks, structured


def chunk_document(
    doc: dict,
    pages: list[dict],
    params: ChunkParams,
    *,
    created_at: str,
) -> tuple[list[dict], list[dict]]:
    """문서 1개의 페이지들을 청킹한다. 반환: (청크 레코드 목록, 경고 목록).

    경고: 구조가 예상과 다른 페이지, 청크가 하나도 나오지 않은 페이지.
    """
    chunks: list[dict] = []
    warnings: list[dict] = []
    book_id = doc["book_id"]
    index = 0
    for page in sorted(pages, key=lambda p: p["page"]):
        if page.get("extraction_status") == "failed" or not page["text"].strip():
            warnings.append({"book_id": book_id, "page": page["page"], "reason": "no_text"})
            continue
        page_chunks, structured = chunk_page(page, params)
        if not structured:
            warnings.append({"book_id": book_id, "page": page["page"], "reason": "unexpected_page_structure"})
        if not page_chunks:
            warnings.append({"book_id": book_id, "page": page["page"], "reason": "no_chunk"})
        for number, c in enumerate(page_chunks, start=1):
            index += 1
            chunks.append({
                "chunk_id": f"{book_id}-p{c['page']}-c{number}",
                "book_id": book_id,
                "file_name": doc["file_name"],
                "page": c["page"],
                "chunk_text": c["chunk_text"],
                "chunk_index": index,
                "chunking_method": CHUNKING_METHOD,
                "chunking_version": CHUNKING_VERSION,
                "dataset_id": doc["dataset_id"],
                "dataset_version": doc["dataset_version"],
                "token_count": None,  # 토크나이저가 정해지는 STEP 3에서 계산한다. 확인 전에는 null
                "metadata": c["metadata"],
                "created_at": created_at,
            })
    return chunks, warnings


def retrieval_text(chunk: dict) -> str:
    """임베딩과 BM25에 넣는 검색용 텍스트: 머리말·문서 제목·소제목을 맥락으로 앞에 붙인다.

    청크 원문(`chunk_text`)은 그대로 두고, 같은 회사·문서를 구분하기 위한 맥락만 더한다.
    """
    meta = chunk.get("metadata") or {}
    prefix = " / ".join(p for p in (meta.get("header"), meta.get("doc_title"), meta.get("section_title")) if p)
    return f"{prefix}\n{chunk['chunk_text']}" if prefix else chunk["chunk_text"]
