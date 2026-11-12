"""확장 브랜치: 질의별 자리 배정 검색이 질문마다 어떤 질의를 만들고 어떻게 재정렬해 최종 12개를 골랐는지 마크다운 한 파일로 보여 준다.

사용: uv run python scripts/export_slots_md_ext.py --out results/_view/slots_IF-DEV.md --ids IF-DEV02 IF-DEV07 ...
(검색·재정렬은 로컬에서 다시 실행하며 LLM 호출은 없다. 분해 질의는 저장된 결과를 읽는다. 정답 쪽 표시는 읽는 사람의 비교용이며 검색에는 쓰이지 않는다.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.config import load_env  # noqa: E402

load_env()

from decompose_queries_ext import load_questions  # noqa: E402
from fact_coverage_ext import load_refs  # noqa: E402
from src.evaluation.retrieval_metrics import reference_pages  # noqa: E402
from src.index.embedding import Embedder  # noqa: E402
from src.index.numpy_store import NumpyIndex  # noqa: E402
from src.retrieval.dense import RetrievalConfig  # noqa: E402
from src.retrieval.dense_np import NumpyDenseRetriever  # noqa: E402
from src.retrieval.hybrid import HybridConfig, HybridRetriever  # noqa: E402
from src.retrieval.rerank import CrossEncoderReranker, RerankConfig, RerankRetriever  # noqa: E402
from src.retrieval.slot_query import allocate_slots  # noqa: E402
from src.retrieval.sparse import SparseRetriever, load_chunks  # noqa: E402

K, POOL, SUB_SLOTS, SUB_CAP, SHOW = 12, 40, 3, 6, 5

HEADER = """# 질의별 자리 배정 검색 — 질문마다 어떻게 12개를 골랐나

## 방식 (한 번에 보기)
1. **질의 만들기**: 원래 질문 1개 + 저장된 분해 질의들(`ext-int-s5-queries-v1`, 프롬프트 query-decompose-v1이 질문만 보고 만든 질의). 분해 질의가 없으면 원래 질문 하나만 쓴다.
2. **질의마다 따로 검색·재정렬**: 질의 하나씩 Dense 20 + BM25 20 → RRF 상위 40 → bge-reranker-v2-m3로 그 **질의**와 점수를 매겨 재정렬한다(질의끼리 합치지 않는다).
3. **자리 배정(총 12개)**: 분해 질의마다 재정렬 상위에서 **질의당 3개**(분해 질의가 3개 이상이면 질의당 2개, 분해 질의 전체 6개 이하)를 먼저 확보한다(이미 뽑힌 청크는 건너뜀). 남은 자리는 **원래 질문의 재정렬 상위**로 채운다.
4. **최종 순서**: 원래 질문에서 채운 청크(순위 순) → 분해 질의에서 확보한 청크. 이 12개가 답변 모델에 간다(문맥 4,000토큰 한도는 그대로).

아래 각 질문에서 ✓ 는 평가 파일이 정답 근거로 지정한 쪽이다(읽는 사람의 비교용, 검색에는 쓰이지 않음).
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", nargs="+", default=["IF-DEV02", "IF-DEV07", "IF-DEV16", "IF-DEV19", "IF-DEV21", "IF-DEV03"])
    ap.add_argument("--queries-file", default="ext-int-s5-queries-v1")
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        print(f"이미 있습니다(덮어쓰지 않음): {out}. (중단)")
        return 1
    refs = load_refs()
    lookup = {}
    for line in (ROOT / "results" / args.queries_file / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            lookup[r["question"]] = r["queries"]
    chunks = load_chunks(ROOT / "data" / "processed_intermediate" / "chunks.jsonl")
    index = NumpyIndex.load("intermediate__chunk-v1__e5large-v1")
    dense = NumpyDenseRetriever(index, RetrievalConfig(), embedder=Embedder(RetrievalConfig().spec))
    hybrid = HybridRetriever(dense, SparseRetriever(chunks=chunks), HybridConfig(dense_candidates=20, sparse_candidates=20, fused_top=POOL, final_k=K))
    rr = RerankRetriever(hybrid, CrossEncoderReranker(), RerankConfig(candidates=POOL, final_k=K))
    qmap = {q["question_id"]: q for q in load_questions()}
    lines = [HEADER]
    for qid in args.ids:
        question = qmap[qid]["question"]
        gold = {(b, p) for b, p in reference_pages(refs[qid]["evidence_groups"])}
        mark = lambda c: " ✓정답쪽" if (c["book_id"], c["page"]) in gold else ""  # noqa: E731
        extras = [x for x in lookup.get(question, []) if x != question]
        original = rr.retrieve(question, k=POOL).chunks
        subs = [rr.retrieve(x, k=POOL).chunks for x in extras]
        final = allocate_slots(original, subs, K, SUB_SLOTS, SUB_CAP)
        lines += [f"---\n\n## {qid}", "", f"**원래 질문**: {question}", "", "**분해 질의**:"] + ([f"{i}. {x}" for i, x in enumerate(extras, 1)] or ["- (없음: 원래 질문만 사용)"]) + [""]
        gold_text = ", ".join(f"{b} p{p}" for b, p in sorted(gold)) or "(없음)"
        lines += [f"**정답 근거 쪽(비교용)**: {gold_text}", ""]
        for label, items in [("원래 질문", original)] + [(f"분해 질의 {i}", s) for i, s in enumerate(subs, 1)]:
            lines += [f"### 재정렬 결과 상위 {SHOW} — {label}", "", "| 순위 | 쪽 | 재정렬 점수 | 비고 |", "|---|---|---|---|"]
            for c in items[:SHOW]:
                lines.append(f"| {c['rank']} | {c['book_id']} p{c['page']} | {c['score']:.3f} | {mark(c).strip()} |")
            first = {}
            for c in items:
                first.setdefault((c["book_id"], c["page"]), c["rank"])
            miss = [f"{b} p{p}={'풀 밖' if (b, p) not in first else str(first[(b, p)]) + '위'}" for b, p in sorted(gold)]
            lines += ["", "정답 쪽 순위(이 질의의 후보 40 안): " + ", ".join(miss), ""]
        lines += ["### 최종 12개", "", "| 순서 | 쪽 | 어디서 뽑았나 | 비고 |", "|---|---|---|---|"]
        for c in final:
            src = "원래 질문 상위" if c["slot_source"] == "original" else f"분해 질의 {c['slot_source'][3:]}에서 확보"
            lines.append(f"| {c['rank']} | {c['book_id']} p{c['page']} | {src} | {mark(c).strip()} |")
        got = sum(1 for g in gold if g in {(c['book_id'], c['page']) for c in final})
        lines += ["", f"→ 정답 근거 쪽 {got}/{len(gold)}개가 최종 12개에 들어옴", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"저장: {out} ({len(args.ids)}개 질문)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
