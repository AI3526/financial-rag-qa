from src.ingest import chunk as ck

PARAMS = ck.ChunkParams()


def _page(body_lines, page=1, header="FB01 | 해온솔루션 경영기획실", title="문서 제목", section="소제목", footer=None):
    lines = [header, title, section, *body_lines, footer or f"{page} / 24"]
    return {"book_id": "FB01", "page": page, "text": "\n".join(lines), "extraction_status": "success"}


DOC = {"book_id": "FB01", "file_name": "FB01.pdf", "dataset_id": "basic", "dataset_version": "basic-v1"}


def test_parse_page_extracts_header_title_section_and_drops_footer():
    meta, body, ok = ck.parse_page(_page(["본문 한 줄"])["text"])
    assert ok
    assert meta == {"header": "FB01 | 해온솔루션 경영기획실", "doc_title": "문서 제목", "section_title": "소제목"}
    assert body == ["본문 한 줄"]  # 꼬리말(1 / 24)은 본문에 없다


def test_parse_page_falls_back_when_structure_is_unexpected():
    meta, body, ok = ck.parse_page("그냥 한 줄\n두 줄")
    assert not ok and meta == {} and body == ["그냥 한 줄", "두 줄"]


def test_paragraphs_split_where_line_is_shorter_than_85_percent_of_longest():
    full = "가" * 50
    lines = [full, full, "짧은 끝줄", full, "둘째 문단 끝"]
    paras = ck.split_paragraph_lines(lines, 0.85)
    assert [len(p) for p in paras] == [3, 2]


def test_join_lines_keeps_words_intact_and_adds_space_only_after_punctuation():
    # 단어 중간에서 끊긴 줄은 공백 없이 이어진다 (회사명이 쪼개지면 안 된다)
    assert ck.join_lines(["해온", "솔루션은 공장"]) == "해온솔루션은 공장"
    # 문장부호로 끝난 줄은 단어 경계이므로 공백을 넣는다
    assert ck.join_lines(["작성했다.", "이 개요는"]) == "작성했다. 이 개요는"
    assert ck.join_lines(["이며,", "운영서비스"]) == "이며, 운영서비스"
    # 다음 줄이 원래 공백으로 시작하면 그 공백을 보존한다
    assert ck.join_lines(["정정된", " 차이를"]) == "정정된 차이를"
    # 문장부호 뒤에 이미 공백이 있으면 공백이 둘이 되지 않는다
    assert ck.join_lines(["끝이다.", " 다음"]) == "끝이다. 다음"


def test_short_paragraphs_are_packed_up_to_max_chars_and_never_exceed():
    p = ck.ChunkParams(max_chars=100)
    full = "가" * 40
    body = [full, "끝1", full, "끝2", full, "끝3"]  # 문단 3개: 각 약 42자
    chunks, ok = ck.chunk_page(_page(body), p)
    assert ok
    assert all(len(c["chunk_text"]) <= 100 for c in chunks)
    assert len(chunks) == 2  # 2개 문단(약 85자)씩 묶이고 나머지 1개
    assert "\n" in chunks[0]["chunk_text"]  # 문단 사이는 줄바꿈


def test_paragraph_over_max_but_within_hard_max_stays_whole():
    p = ck.ChunkParams(max_chars=60, hard_max_chars=200)
    sentence = "이것은 문장 하나의 길이를 채우기 위한 문장이다."
    long_para = sentence + " " + sentence + " " + sentence  # 약 95자: max_chars 초과, hard_max 이내
    body = ["가" * 20 + "끝", long_para]
    chunks, _ = ck.chunk_page(_page(body), p)
    assert len(chunks) == 2  # 긴 문단은 이웃과 묶이지 않고 통째로 하나의 청크
    assert chunks[1]["chunk_text"] == long_para  # 문장 중간에서 갈라지지 않는다


def test_paragraph_over_hard_max_is_split_by_sentence_without_loss():
    p = ck.ChunkParams(max_chars=60, hard_max_chars=60)
    sentence = "이것은 문장 하나의 길이를 채우기 위한 문장이다."
    body = [sentence + " " + sentence + " " + sentence]  # 한 줄이 곧 한 문단, 한도 초과
    chunks, _ = ck.chunk_page(_page(body), p)
    assert len(chunks) >= 2
    assert all(len(c["chunk_text"]) <= 60 for c in chunks)
    joined = " ".join(c["chunk_text"] for c in chunks)
    assert joined.count("이것은") == 3  # 내용 손실 없음


def test_overlap_repeats_last_unit_in_next_chunk():
    p = ck.ChunkParams(max_chars=100, overlap_units=1)
    full = "가" * 40
    body = [full, "끝1", full, "끝2", full, "끝3"]
    chunks, _ = ck.chunk_page(_page(body), p)
    assert len(chunks) >= 2
    last_of_first = chunks[0]["chunk_text"].split("\n")[-1]
    assert chunks[1]["chunk_text"].split("\n")[0] == last_of_first


def test_chunk_document_ids_pages_and_no_page_crossing():
    pages = [_page(["가" * 30, "끝"], page=1), _page(["나" * 30, "끝"], page=2)]
    chunks, warnings = ck.chunk_document(DOC, pages, PARAMS, created_at="2026-01-01T00:00:00+00:00")
    assert warnings == []
    assert [c["chunk_id"] for c in chunks] == ["FB01-p1-c1", "FB01-p2-c1"]
    assert [c["page"] for c in chunks] == [1, 2]
    assert [c["chunk_index"] for c in chunks] == [1, 2]
    assert "가" in chunks[0]["chunk_text"] and "나" not in chunks[0]["chunk_text"]  # 페이지를 넘지 않는다
    assert set(chunks[0]) == {
        "chunk_id", "book_id", "file_name", "page", "chunk_text", "chunk_index", "chunking_method",
        "chunking_version", "dataset_id", "dataset_version", "token_count", "metadata", "created_at",
    }
    assert chunks[0]["token_count"] is None  # 확인하지 못한 값은 0이 아니라 null
    assert chunks[0]["chunking_version"] == "chunk-v1"


def test_chunk_document_records_warnings_for_empty_and_unstructured_pages():
    empty = {"book_id": "FB01", "page": 1, "text": "   ", "extraction_status": "partial"}
    odd = {"book_id": "FB01", "page": 2, "text": "머리말 없는 한 줄", "extraction_status": "success"}
    chunks, warnings = ck.chunk_document(DOC, [empty, odd], PARAMS, created_at="t")
    reasons = {(w["page"], w["reason"]) for w in warnings}
    assert (1, "no_text") in reasons and (2, "unexpected_page_structure") in reasons
    assert len(chunks) == 1 and chunks[0]["page"] == 2  # 구조가 달라도 내용은 버리지 않는다


def test_retrieval_text_prefixes_context_without_changing_chunk_text():
    chunks, _ = ck.chunk_page(_page(["가" * 20, "끝"]), PARAMS)
    c = {**DOC, **chunks[0]}
    rt = ck.retrieval_text(c)
    assert rt.startswith("FB01 | 해온솔루션 경영기획실 / 문서 제목 / 소제목\n")
    assert rt.endswith(c["chunk_text"])
    assert "1 / 24" not in c["chunk_text"]
    assert ck.retrieval_text({"chunk_text": "본문", "metadata": {}}) == "본문"


def test_chunking_is_deterministic():
    page = _page(["가" * 50, "끝", "나" * 50, "끝"])
    a, _ = ck.chunk_page(page, PARAMS)
    b, _ = ck.chunk_page(page, PARAMS)
    assert a == b


def test_punct_join_keeps_wrapped_prose_lines_together_but_default_is_unchanged():
    # 최장 줄(표 줄)이 기준을 올린 쪽: 기본 규칙은 본문 줄마다 문단을 끊고, v2f는 문장부호로 끝나지 않은 본문 줄을 다음 줄과 잇는다
    long_line = "표 줄 " + "가" * 66
    prose0 = "본문은 줄마다 폭이 일정하게 나뉘어 있어서 한 문장이 여러 줄에 걸쳐 이어" + "진다"
    prose1 = "서 쓰인 경우에도 문장 중간에서 문단이 끊기지 않아야 한다."
    lines = ["제목", long_line, prose0, prose1, "표 칸"]
    default = ck.split_paragraph_lines(lines, 0.85)
    joined = ck.split_paragraph_lines(lines, 0.85, punct_join=True, prose_min_chars=40)
    same_paragraph = lambda paras: any(prose0 in p and prose1 in p for p in paras)  # noqa: E731
    assert not same_paragraph(default)  # 기본: prose0 뒤에서 문단이 끊김(문장 중간)
    assert same_paragraph(joined)  # v2f: 한 문단
    assert ["표 칸"] in joined  # 짧은 줄(표 칸)은 이전처럼 문단의 끝
    assert ck.ChunkParams().as_dict() == ck.ChunkParams(punct_join=False).as_dict()
