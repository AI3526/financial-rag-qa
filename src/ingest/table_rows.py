"""표 행 문장화 (확장 브랜치 3단계, EXTENSION_INTERMEDIATE_LOG).

표의 각 행을 머리줄의 열 이름과 합쳐 그 행만 봐도 의미가 완결되는 문장으로 바꾼다. 예:
    머리줄 [항목, 예산(백만원), 집행(백만원)] + 행 [CAP-DP-06, 3500, 1200]
    → "CAP-DP-06: 예산(백만원) 3500, 집행(백만원) 1200."
- 이 모듈은 표 구조(셀 목록)를 입력으로 받는 순수 함수다. 셀 목록은 pdfplumber 같은 도구가 읽는다(`scripts/build_table_chunks_ext.py`).
- 첫 행을 머리줄로 본다(Intermediate 전체 표에서 확인한 구조). 첫 열은 행 이름으로 쓰고 나머지 열은 `열 이름 값`으로 잇는다.
- 값이 비어 있는 셀은 건너뛰고, 줄바꿈이 든 셀은 공백으로 잇는다. 숫자와 단위는 원문 그대로 둔다(변환·계산 없음).
- 문장화한 텍스트는 검색용 보조 청크이며 인용과 답변의 근거 원문은 원래 청크와 쪽이다.
"""
from __future__ import annotations

import re

_NUMERIC = re.compile(r"[-+]?[\d,\.]+%?")


def _is_hangul(ch: str) -> bool:
    return "가" <= ch <= "힣"


def clean_cell(cell) -> str:
    """셀 텍스트를 한 줄로 만든다. 칸 폭 때문에 줄바꿈된 곳은 앞뒤 글자가 모두 한글이면 단어 중간에서 끊긴 것으로 보고 공백 없이 잇고
    (예: `루미뷰 정`+`기` → `루미뷰 정기`), 그 밖에는 공백 하나로 잇는다(예: `루미센서`+`S` → `루미센서 S`)."""
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in (cell or "").split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""
    out = lines[0]
    for nxt in lines[1:]:
        out += nxt if _is_hangul(out[-1]) and _is_hangul(nxt[0]) else " " + nxt
    return out


def is_header_like(cells: list[str]) -> bool:
    """머리줄처럼 보이는가: 비어 있지 않은 셀이 하나 이상 있고 모두 숫자만으로 된 값이 아니다."""
    filled = [c for c in cells if c]
    return bool(filled) and not any(_NUMERIC.fullmatch(c) for c in filled)


def serialize_table(table_rows: list[list]) -> list[str]:
    """표(셀 목록의 목록)를 행 문장 목록으로 바꾼다. 데이터 행이 없으면 빈 목록."""
    rows = [[clean_cell(c) for c in r] for r in table_rows]
    rows = [r for r in rows if any(r)]
    if len(rows) < 2:
        return []
    header, sentences = rows[0], []
    for row in rows[1:]:
        key = row[0] if row else ""
        parts = []
        for name, value in zip(header[1:], row[1:]):
            if value:
                parts.append(f"{name} {value}" if name else value)
        if key and parts:
            sentences.append(f"{key}: " + ", ".join(parts) + ".")
        elif key:
            sentences.append(f"{key}.")
        elif parts:
            sentences.append(", ".join(parts) + ".")
    return sentences


def pack_sentences(sentences: list[str], max_chars: int = 500) -> list[str]:
    """행 문장을 줄바꿈으로 이어 max_chars 이하의 청크 본문으로 묶는다(한 문장이 길어도 쪼개지 않는다)."""
    out: list[str] = []
    cur: list[str] = []
    for s in sentences:
        if cur and len("\n".join(cur + [s])) > max_chars:
            out.append("\n".join(cur))
            cur = []
        cur.append(s)
    if cur:
        out.append("\n".join(cur))
    return out
