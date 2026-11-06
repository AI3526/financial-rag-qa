"""실제 Basic PDF 15개에 대한 STEP 1 검증 (API 호출 없음, 약 20초).

- 전체 문서·페이지 수, 실패 없음
- 꼬리말 `N / 24`가 PDF 페이지 번호와 일치하는지 (페이지 번호 연결 정확성)
- 같은 입력이면 같은 sha256
"""
import re
from pathlib import Path

import pytest

from src.ingest import extract as ex

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "data" / "finance_basic" / "documents"

pytestmark = pytest.mark.skipif(not DOCS_DIR.exists(), reason="데이터 폴더가 없습니다")


@pytest.fixture(scope="module")
def extracted():
    docs, pages = [], []
    for pdf in sorted(DOCS_DIR.glob("*.pdf")):
        d, p = ex.extract_pdf(pdf, repo_root=ROOT, dataset_id="basic", dataset_version="basic-v1")
        docs.append(d)
        pages.extend(p)
    return docs, pages


def test_all_15_documents_360_pages_no_failures(extracted):
    docs, pages = extracted
    assert len(docs) == 15
    assert len(pages) == 360
    assert [d["book_id"] for d in docs] == [f"FB{n:02d}" for n in range(1, 16)]
    assert all(d["extraction_status"] == "success" for d in docs), [
        (d["book_id"], d["error_message"]) for d in docs if d["extraction_status"] != "success"
    ]
    assert all(d["page_count"] == 24 for d in docs)
    assert all(p["text"].strip() for p in pages)


def test_footer_page_number_matches_pdf_page_index(extracted):
    _, pages = extracted
    mismatches = []
    for p in pages:
        last = p["text"].strip().split("\n")[-1].strip()
        m = re.fullmatch(r"(\d+) / (\d+)", last)
        if not m or int(m.group(1)) != p["page"] or int(m.group(2)) != 24:
            mismatches.append((p["book_id"], p["page"], last))
    assert mismatches == []


def test_sha256_is_deterministic_and_text_is_raw(extracted):
    docs, pages = extracted
    first = DOCS_DIR / "FB01.pdf"
    assert ex.sha256_file(first) == docs[0]["file_sha256"]
    # 가공하지 않은 원문: 머리말이 그대로 남아 있다
    fb01_p1 = next(p for p in pages if p["book_id"] == "FB01" and p["page"] == 1)
    assert fb01_p1["text"].startswith("FB01 | 해온솔루션 경영기획실")
