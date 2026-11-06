"""실제 Basic 데이터(`data/processed/pages.jsonl`, `documents.jsonl`)에 대한 STEP 2 검증 (오프라인).

- 360쪽 모두 청킹되고 경고가 없다
- 청크는 한도를 넘지 않고 페이지 경계를 넘지 않는다
- 페이지별로 내용 손실도 중복도 없다 (overlap 0)
- 머리말·꼬리말이 본문에 섞이지 않고 회사·제품명이 쪼개지지 않는다
- development 근거 원문(quote)이 청크 하나 안에 온전히 들어 있다 (청킹 충실도 점검용, 튜닝에는 쓰지 않음)
"""
import json
import re
import unicodedata
from pathlib import Path

import pytest

from src.ingest import chunk as ck

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "data" / "processed" / "pages.jsonl"
DOCS = ROOT / "data" / "processed" / "documents.jsonl"
DEV = ROOT / "data" / "finance_basic" / "evaluation" / "development.json"

pytestmark = pytest.mark.skipif(not (PAGES.exists() and DOCS.exists()), reason="STEP 1 결과(data/processed)가 없습니다")


def _read(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _norm(s):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))


@pytest.fixture(scope="module")
def result():
    docs, pages = _read(DOCS), _read(PAGES)
    params = ck.ChunkParams()
    chunks, warnings = [], []
    for doc in docs:
        c, w = ck.chunk_document(doc, [p for p in pages if p["book_id"] == doc["book_id"]], params,
                                 created_at="2026-01-01T00:00:00+00:00")
        chunks += c
        warnings += w
    return docs, pages, chunks, warnings, params


def test_all_pages_chunked_without_warnings(result):
    docs, pages, chunks, warnings, _ = result
    assert len(docs) == 15 and len(pages) == 360
    assert warnings == []
    assert {(c["book_id"], c["page"]) for c in chunks} == {(p["book_id"], p["page"]) for p in pages}


def test_chunk_size_limit_and_unique_ids(result):
    _, _, chunks, _, params = result
    assert all(0 < len(c["chunk_text"]) <= params.hard_max_chars for c in chunks)
    # max_chars를 넘는 청크는 쪼개지 않은 문단 하나여야 한다 (여러 문단을 묶은 청크는 max_chars 이내)
    assert all("\n" not in c["chunk_text"] for c in chunks if len(c["chunk_text"]) > params.max_chars)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"FB\d{2}-p\d+-c\d+", i) for i in ids)


def test_no_loss_no_duplication_and_no_page_crossing(result):
    _, pages, chunks, _, _ = result
    for p in pages:
        _, body, _ = ck.parse_page(p["text"])
        page_chunks = [c for c in chunks if c["book_id"] == p["book_id"] and c["page"] == p["page"]]
        assert _norm("".join(c["chunk_text"] for c in page_chunks)) == _norm("".join(body)), (p["book_id"], p["page"])


def test_header_footer_excluded_and_kept_in_metadata(result):
    _, _, chunks, _, _ = result
    for c in chunks:
        for line in c["chunk_text"].split("\n"):
            assert not re.fullmatch(r"\d+ / \d+", line.strip())
        assert not c["chunk_text"].startswith(c["book_id"])
        assert set(c["metadata"]) == {"header", "doc_title", "section_title"}
        assert c["metadata"]["header"].startswith(c["book_id"])
        assert ck.retrieval_text(c).startswith(c["metadata"]["header"])


def test_company_and_product_names_are_not_split(result):
    _, _, chunks, _, _ = result
    bad = re.compile(r"해\s온\s?솔루션|해온\s솔루션|해\s온\s?소재|해온\s소재|리브\s필름|리브\s리턴|루미\s센서|루미\s뷰")
    assert [c["chunk_id"] for c in chunks if bad.search(c["chunk_text"])] == []


def test_development_evidence_quotes_fit_inside_one_chunk(result):
    _, _, chunks, _, _ = result
    dev = json.loads(DEV.read_text(encoding="utf-8"))
    quotes = [(e["book_id"], e["page"], e["quote"]) for i in dev for g in i["evidence_groups"] for e in g]
    missing = []
    for book, page, quote in quotes:
        page_chunks = [c for c in chunks if c["book_id"] == book and c["page"] == page]
        if not any(_norm(quote) in _norm(c["chunk_text"]) for c in page_chunks):
            missing.append((book, page))
    assert len(quotes) == 64
    assert missing == []
