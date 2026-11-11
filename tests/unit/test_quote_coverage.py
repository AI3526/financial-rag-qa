"""정답 근거 문장이 최종 청크 안에 있는지 재는 지표 (확장 브랜치 4단계)."""
from src.evaluation.ranking_metrics import quote_coverage


def _ev(*quotes):
    return [[{"book_id": "B", "page": 1, "quote": q}] for q in quotes]


def test_counts_quotes_found_inside_a_single_chunk_ignoring_whitespace():
    chunks = [{"text": "해온소재의 2026년 상반기 별도 매출은 860억원, 영업이익은 60억원이다."}, {"text": "무관한 문장"}]
    assert quote_coverage(chunks, _ev("해온소재의 2026년 상반기 별도 매출은 860억원, 영업이익은 60억원이다.")) == (1, 1.0)
    assert quote_coverage([{"text": "해온소재의2026년상반기별도매출은860억원,영업이익은60억원이다."}], _ev("해온소재의 2026년 상반기 별도 매출은 860억원, 영업이익은 60억원이다.")) == (1, 1.0)


def test_partial_and_missing_coverage():
    chunks = [{"text": "첫째 문장입니다."}]
    assert quote_coverage(chunks, _ev("첫째 문장입니다.", "둘째 문장입니다.")) == (1, 0.5)
    assert quote_coverage(chunks, _ev("없는 문장")) == (0, 0.0)


def test_no_quotes_is_not_applicable_never_zero():
    assert quote_coverage([{"text": "x"}], []) == (None, None)
    assert quote_coverage([{"text": "x"}], [[{"book_id": "B", "page": 1}]]) == (None, None)


def test_a_quote_split_across_two_chunks_is_not_counted():
    assert quote_coverage([{"text": "앞부분 문장"}, {"text": "뒷부분 문장"}], _ev("앞부분 문장 뒷부분 문장")) == (0, 0.0)
