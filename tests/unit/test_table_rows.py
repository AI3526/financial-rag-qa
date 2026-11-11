"""표 행 문장화 순수 함수 (확장 브랜치 3단계). pdfplumber 없이 확인한다."""
from src.ingest import table_rows as tr


def test_serialize_joins_row_key_with_column_names_and_keeps_units_and_numbers_verbatim():
    rows = [["항목", "예산(백만원)", "집행(백만원)", "가용(백만원)"], ["CAP-DP-06", "3500", "1200", "1500"], ["CAP-DP-01", "1,309", "1003", "138"]]
    assert tr.serialize_table(rows) == ["CAP-DP-06: 예산(백만원) 3500, 집행(백만원) 1200, 가용(백만원) 1500.",
                                        "CAP-DP-01: 예산(백만원) 1,309, 집행(백만원) 1003, 가용(백만원) 138."]


def test_multiline_cells_become_one_line_and_empty_cells_are_skipped():
    rows = [["계약잔고번호", "법인·제품", "잔고(백만원)", "후속 상태"], ["HS-BK-0001", "해온솔루션 루미센서\nS", "138", None], ["", "", "", ""]]
    assert tr.serialize_table(rows) == ["HS-BK-0001: 법인·제품 해온솔루션 루미센서 S, 잔고(백만원) 138."]


def test_clean_cell_joins_wrapped_hangul_without_space_but_keeps_space_before_latin():
    assert tr.clean_cell("루미뷰 정\n기") == "루미뷰 정기"
    assert tr.clean_cell("해온솔루션 루미센서\nS") == "해온솔루션 루미센서 S"
    assert tr.clean_cell("  가   나 ") == "가 나" and tr.clean_cell(None) == "" and tr.clean_cell("\n \n") == ""


def test_table_without_data_rows_gives_nothing():
    assert tr.serialize_table([["항목", "금액"]]) == [] and tr.serialize_table([]) == []


def test_two_column_table_and_missing_header_name():
    assert tr.serialize_table([["항목", "금액(억원)"], ["매출", "1,160"]]) == ["매출: 금액(억원) 1,160."]
    assert tr.serialize_table([["항목", ""], ["매출", "1,160"]]) == ["매출: 1,160."]


def test_is_header_like_rejects_numeric_rows():
    assert tr.is_header_like(["항목", "예산(백만원)"]) and not tr.is_header_like(["CAP-DP-01", "1309", "1,003"])
    assert not tr.is_header_like(["", ""])


def test_pack_sentences_respects_max_chars_without_splitting_a_sentence():
    s = ["가" * 200, "나" * 200, "다" * 200]
    packed = tr.pack_sentences(s, max_chars=450)
    assert packed == ["가" * 200 + "\n" + "나" * 200, "다" * 200]
    assert tr.pack_sentences(["가" * 900], max_chars=450) == ["가" * 900]  # 긴 문장 하나는 그대로 둔다
