import hashlib

from src.generation import context as cx
from src.generation.prompts import PROMPTS_DIR, load_prompt


def chunk(i, text, book="FB01", page=1):
    return {"rank": i, "score": 0.9, "chunk_id": f"{book}-p{page}-c{i}", "book_id": book,
            "file_name": f"{book}.pdf", "page": page, "text": text, "metadata": {}}


def cost(text, n):
    return cx.estimate_tokens(f"[S{n}] {text}") + 2


def test_build_sources_numbers_from_s1_and_keeps_document_and_page():
    s = cx.build_sources([chunk(1, "가", "FB03", 7), chunk(2, "나", "FB09", 2)])
    assert [x["source_id"] for x in s] == ["S1", "S2"]
    assert (s[1]["book_id"], s[1]["file_name"], s[1]["page"], s[1]["chunk_id"]) == ("FB09", "FB09.pdf", 2, "FB09-p2-c2")


def test_everything_fits_when_under_the_limit():
    chunks = [chunk(i, "매출은 1,280억원이다. " * 3) for i in range(1, 6)]
    ctx = cx.build_final_context(chunks, 4000)
    assert len(ctx.sources) == 5 and ctx.dropped_chunk_ids == [] and not ctx.over_limit
    assert 0 < ctx.token_count <= 4000


def test_lowest_ranked_chunks_are_dropped_whole_and_ids_are_renumbered():
    texts = ["첫째 청크 내용입니다. " * 6, "둘째 청크 내용입니다. " * 6, "셋째 청크 내용입니다. " * 6]
    limit = cost(texts[0], 1) + cost(texts[1], 2)  # 앞의 둘만 들어간다
    ctx = cx.build_final_context([chunk(i, t) for i, t in enumerate(texts, 1)], limit)
    assert [s["chunk_id"] for s in ctx.sources] == ["FB01-p1-c1", "FB01-p1-c2"]
    assert ctx.dropped_chunk_ids == ["FB01-p1-c3"]
    assert [s["source_id"] for s in ctx.sources] == ["S1", "S2"]
    assert ctx.sources[0]["text"] == texts[0]  # 청크를 자르지 않는다
    assert ctx.token_count <= limit


def test_a_later_small_chunk_never_jumps_ahead_of_a_dropped_larger_one():
    big, small = "큰 청크 내용입니다. " * 30, "작음"
    ctx = cx.build_final_context([chunk(1, "첫째 " * 4), chunk(2, big), chunk(3, small)], cost("첫째 " * 4, 1) + 5)
    assert [s["chunk_id"] for s in ctx.sources] == ["FB01-p1-c1"]  # 순위를 건너뛰지 않는다
    assert ctx.dropped_chunk_ids == ["FB01-p1-c2", "FB01-p1-c3"]


def test_first_chunk_larger_than_limit_is_kept_whole_and_flagged():
    ctx = cx.build_final_context([chunk(1, "아주 긴 청크입니다. " * 50), chunk(2, "다음")], 20)
    assert len(ctx.sources) == 1 and ctx.over_limit and ctx.sources[0]["text"].startswith("아주 긴")
    assert ctx.dropped_chunk_ids == ["FB01-p1-c2"]


def test_empty_retrieval_gives_empty_context_text():
    ctx = cx.build_final_context([], 4000)
    assert ctx.sources == [] and ctx.token_count == 0 and cx.format_context(ctx.sources) == cx.NO_CONTEXT_TEXT


def test_record_is_labeled_as_estimate():
    rec = cx.build_final_context([chunk(1, "내용")], 4000).as_record()
    assert rec["token_count_method"].endswith("(estimate)") and rec["context_limit_tokens"] == 4000


def test_prompt_file_hash_and_version_are_recorded():
    spec = load_prompt()
    assert spec.version == "answer-v2"  # 현재 기본 프롬프트(STEP 9 조정 결과)
    assert spec.sha256 == hashlib.sha256((PROMPTS_DIR / "answer_v2.json").read_bytes()).hexdigest()
    assert "{context}" in spec.user and "{question}" in spec.user
    assert "expected_action" not in spec.system + spec.user and "reference_answer" not in spec.system + spec.user


def test_every_prompt_file_has_a_unique_version_matching_its_name_and_no_evaluation_terms():
    import json

    versions = []
    for path in sorted(PROMPTS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        # 파일 이름 `answer_v2.json`, `faithfulness_statement_v1.json` -> 버전 `answer-v2`, `faithfulness-statement-v1`
        assert data["version"] == path.stem.replace("_", "-")
        assert data["type"] in ("answer", "evaluation")
        if data["type"] == "answer":  # 서비스 답변 프롬프트는 검색 자료와 질문 자리가 있는 시스템·사용자 메시지다
            assert "{context}" in data["user"] and "{question}" in data["user"]
            text = data["system"] + data["user"]
        else:  # 평가 프롬프트(ragas 1단계 등)는 지시문과 예시를 가진다
            text = json.dumps(data, ensure_ascii=False)
        # 서비스 프롬프트에는 평가용 정보가 들어가면 안 된다. 평가 프롬프트도 기본은 정답 필드 이름을 쓰지 않으며,
        # 참고 정답을 입력받는 Judge만 `may_use_reference_data: true`로 명시해 예외로 둔다(그 외 모든 프롬프트는 계속 검사한다)
        allowed = data.get("may_use_reference_data", False)
        assert not allowed or data["type"] == "evaluation"
        for forbidden in ("reference_answer", "expected_action", "evidence_groups", "required_facts", "forbidden_claims"):
            if not allowed:
                assert forbidden not in text
        versions.append(data["version"])
    assert len(versions) == len(set(versions)) and {"answer-v1", "answer-v2", "faithfulness-statement-v1", "judge-v1", "judge-v2"} <= set(versions)


def test_v2_differs_from_v1_only_in_rule_4_so_the_change_is_isolated():
    import json

    v1 = json.loads((PROMPTS_DIR / "answer_v1.json").read_text(encoding="utf-8"))
    v2 = json.loads((PROMPTS_DIR / "answer_v2.json").read_text(encoding="utf-8"))
    a, b = v1["system"].split("\n"), v2["system"].split("\n")
    changed = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert len(a) == len(b) and len(changed) == 1 and b[changed[0]].startswith("4. ")
    assert v1["user"] == v2["user"]
