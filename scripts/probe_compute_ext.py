"""확장 브랜치 진단(상한선): 사람이 분해한 질의의 검색 결과(시험 A)와 사실 목록(시험 B)을 문맥으로 주고 기존 답변 프롬프트(answer_v2) 그대로 계산형 질문에 답하게 한다.

- 새 프롬프트 없음(기존 answer_v2), 문맥 한도 4,000토큰 그대로. 반복 횟수만큼 같은 입력으로 호출한다(생성 변동 확인).
- 시험 A: 질의마다 Hybrid(Dense 20 + BM25 20 → RRF) 상위 `--per-query`개 청크를 모아(중복 제거) 문맥으로 쓴다(재정렬 없음).
- 시험 B(진단용): 질의 파일의 `fact_sheet`(값, 단위, 기간, 범위, 조건 문장, 출처 쪽을 담은 사실 목록; 평가 정답이 아니라 원문 쪽 텍스트에서 옮겨 적은 것)를 출처 하나씩으로 준다.
- 정답을 아는 사람이 만든 질의·사실 목록이라 방법의 성능 평가가 아니라 '계산 단계가 되는가'를 보는 진단이다.

사용: uv run python scripts/probe_compute_ext.py --spec data/finance_intermediate/evaluation_ext/handmade_IF-DEV19.json --repeats 3 --out results/_view/IF-DEV19_compute_probe.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_env  # noqa: E402

load_env()

from src.api.pipeline import PipelineSettings  # noqa: E402
from src.generation.context import build_final_context, build_sources  # noqa: E402
from src.generation.generator import LangChainGenerator  # noqa: E402
from src.generation.prompts import load_prompt  # noqa: E402
from src.index.embedding import Embedder  # noqa: E402
from src.index.numpy_store import NumpyIndex  # noqa: E402
from src.retrieval.dense import RetrievalConfig  # noqa: E402
from src.retrieval.dense_np import NumpyDenseRetriever  # noqa: E402
from src.retrieval.hybrid import HybridConfig, HybridRetriever  # noqa: E402
from src.retrieval.sparse import SparseRetriever, load_chunks  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--per-query", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        print(f"이미 있습니다(덮어쓰지 않음): {out}. (중단)")
        return 1
    spec = json.loads((ROOT / args.spec).read_text(encoding="utf-8"))
    chunks = load_chunks(ROOT / "data" / "processed_intermediate" / "chunks.jsonl")
    index = NumpyIndex.load("intermediate__chunk-v1__e5large-v1")
    dense = NumpyDenseRetriever(index, RetrievalConfig(), embedder=Embedder(RetrievalConfig().spec))
    hybrid = HybridRetriever(dense, SparseRetriever(chunks=chunks), HybridConfig(dense_candidates=20, sparse_candidates=20, fused_top=40, final_k=5))
    settings = PipelineSettings()
    generator = LangChainGenerator(settings, prompt=load_prompt(), callbacks_factory=None)
    # 시험 A: 질의별 Hybrid 상위 청크
    picked, seen = [], set()
    for item in spec["queries"]:
        n = 0
        for c in hybrid.retrieve(item["query"], k=40).chunks:
            if n >= args.per_query:
                break
            if c["chunk_id"] in seen:
                continue
            seen.add(c["chunk_id"])
            picked.append({**c, "rank": len(picked) + 1})
            n += 1
    ctx_a = build_final_context(picked, settings.context_limit_tokens)
    # 시험 B: 사실 목록(출처 하나 = 사실 한 줄)
    fake = [{"chunk_id": f"fact-{i}", "book_id": f["book_id"], "file_name": f"{f['book_id']}.pdf", "page": f["page"], "rank": i, "text": f["text"]}
            for i, f in enumerate(spec["fact_sheet"], start=1)]
    ctx_b = build_final_context(fake, settings.context_limit_tokens)
    lines = [f"# {spec['question_id']} 계산형 질문 시험 (기존 프롬프트 answer_v2, 새 프롬프트 없음)", "", f"**질문**: {spec['question']}", "",
             "정답 기준(평가 파일의 필수 사실, 비교용): Q1 22,000/60,000≈36.67%, Q2 26,000/68,000≈38.24%, 분모가 달라 단순 평균 불가, 상반기 48,000/128,000=37.50%", ""]
    summary = {}
    for label, ctx in (("A 질의별 검색 결과(쪽 텍스트)", ctx_a), ("B 사실 목록", ctx_b)):
        lines += [f"## 시험 {label}", "", "**문맥에 들어간 출처**: " + ", ".join(f"[{s['source_id']}] {s['book_id']} p{s['page']}" for s in ctx.sources) + f" (추정 {ctx.token_count}토큰)", ""]
        for i in range(1, args.repeats + 1):
            res = generator.generate(spec["question"], ctx.sources)
            p = res.parsed or {}
            summary.setdefault(label, []).append(p.get("status"))
            lines += [f"### {label} — 시도 {i}", f"- 상태: **{p.get('status')}**", f"- 답변: {p.get('answer')}", f"- 이유: {p.get('reason')}", f"- 출처: {p.get('source_ids')}", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n저장: {out}\n상태 요약: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
