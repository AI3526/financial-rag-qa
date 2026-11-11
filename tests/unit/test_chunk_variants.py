"""확장 실험용 청킹 옵션(꼬리 병합, 표 덩어리 유지)과 기본값 불변 (extension/intermediate)."""
from src.ingest import chunk as ck


def _page(body_lines, page=1):
    lines = ["IF39 | 다온정밀 자료관리실", "다온정밀 투자집행 관리대장", "항목별 관리대장", *body_lines, f"{page} / 90"]
    return {"book_id": "IF39", "page": page, "text": "\n".join(lines), "extraction_status": "success"}


DOC = {"book_id": "IF39", "file_name": "IF39.pdf", "dataset_id": "intermediate", "dataset_version": "intermediate-v1"}
LONG = "가" * 58  # 페이지 최장 줄(문단 판정 기준)


def _table_page():
    prose = [LONG, LONG, "본문 끝줄입니다."]
    table = ["항목 예산(백만원) 집행(백만원) 잔여약정(백만원)", "CAP-DP-01 1309 1003 168", "CAP-DP-02 1079 638 372", "CAP-DP-03 1527 1140 223",
             "CAP-DP-04 1298 775 427"]
    return _page(prose + table + [LONG, LONG, "마지막 문단 끝줄"])


def test_default_params_dict_is_unchanged_so_chunk_v1_manifests_stay_identical():
    assert set(ck.ChunkParams().as_dict()) == {"max_chars", "hard_max_chars", "overlap_units", "paragraph_ratio", "join_rule"}
    assert "min_chunk_chars" in ck.ChunkParams(min_chunk_chars=60).as_dict() and "table_block" in ck.ChunkParams(table_block=True).as_dict()


def test_group_table_blocks_merges_only_runs_of_three_or_more_short_paragraphs():
    long = "나" * 120
    assert ck.group_table_blocks(["a", "b", long, "c", "d", "e", "f"]) == ["a", "b", long, "c\nd\ne\nf"]
    assert ck.group_table_blocks(["a", "b"]) == ["a", "b"]


def test_split_block_repeats_the_first_row_in_every_piece():
    pieces = ck._split_block("머리줄\n행1 111\n행2 222\n행3 333", max_chars=14)
    assert len(pieces) >= 2 and all(p.startswith("머리줄\n") for p in pieces)
    assert "".join(p.split("\n", 1)[1] for p in pieces).count("행") == 3  # 행이 빠지거나 중복되지 않는다


def test_table_block_keeps_header_and_all_rows_in_one_chunk_even_with_small_max_chars():
    plain, _ = ck.chunk_page(_table_page(), ck.ChunkParams(max_chars=100))
    kept, _ = ck.chunk_page(_table_page(), ck.ChunkParams(max_chars=100, table_block=True))
    header = "항목 예산(백만원)"
    plain_with_header = [c for c in plain if header in c["chunk_text"]][0]
    kept_with_header = [c for c in kept if header in c["chunk_text"]][0]
    assert not all(f"CAP-DP-0{i}" in plain_with_header["chunk_text"] for i in range(1, 5))  # 기존 방식은 머리줄과 행이 갈린다
    assert all(f"CAP-DP-0{i}" in kept_with_header["chunk_text"] for i in range(1, 5))  # 표 덩어리는 한 청크에 남는다


def test_merge_small_groups_joins_a_short_tail_into_the_previous_chunk_within_hard_max():
    p = ck.ChunkParams(max_chars=50, hard_max_chars=200, min_chunk_chars=10)
    groups = [[("가" * 45, 0)], [("끝.", 0)]]
    merged = ck.merge_small_groups(groups, p)
    assert len(merged) == 1 and ck._render(merged[0]).endswith("끝.")
    full = [[("가" * 45, 0)], [("끝.", 1)]]
    assert len(ck.merge_small_groups(full, ck.ChunkParams(max_chars=50, hard_max_chars=46, min_chunk_chars=10))) == 2  # hard_max를 넘으면 합치지 않는다
    assert len(ck.merge_small_groups([[("끝.", 0)], [("가" * 45, 1)]], p)) == 2  # 첫 청크는 합치지 않는다


def test_chunk_document_records_the_given_version_and_defaults_to_chunk_v1():
    page = _table_page()
    default, _ = ck.chunk_document(DOC, [page], ck.ChunkParams(), created_at="t")
    tagged, _ = ck.chunk_document(DOC, [page], ck.ChunkParams(table_block=True), created_at="t", version="chunk-v2c-tableblock")
    assert {c["chunking_version"] for c in default} == {"chunk-v1"} and {c["chunking_version"] for c in tagged} == {"chunk-v2c-tableblock"}


def test_default_chunking_ignores_new_options():
    page = _table_page()
    a, _ = ck.chunk_page(page, ck.ChunkParams())
    b, _ = ck.chunk_page(page, ck.ChunkParams(min_chunk_chars=0, table_block=False))
    assert a == b
