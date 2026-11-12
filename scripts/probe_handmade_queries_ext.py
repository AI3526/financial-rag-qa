"""확장 브랜치 진단(상한선 확인): 사람이 분해해 쓴 질의로 단순 Dense·Hybrid 검색을 했을 때 정답 쪽을 찾는지 본다 (재정렬·LLM 호출 없음).

질의와 정답 쪽은 JSON 파일(`--spec`)에서 읽는다(문구를 고치지 않는다). 정답 쪽을 아는 사람이 쓴 질의이므로 방법의 성능 평가가 아니라 '이렇게 분해해 주면 찾을 수 있는가'를 보는 진단이다.
사용: uv run python scripts/probe_handmade_queries_ext.py --spec data/finance_intermediate/evaluation_ext/handmade_IF-DEV19.json --out results/_view/IF-DEV19_handmade.md
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

from src.index.embedding import Embedder  # noqa: E402
from src.index.numpy_store import NumpyIndex  # noqa: E402
from src.retrieval.dense import RetrievalConfig  # noqa: E402
from src.retrieval.dense_np import NumpyDenseRetriever  # noqa: E402
from src.retrieval.hybrid import HybridConfig, HybridRetriever  # noqa: E402
from src.retrieval.sparse import SparseRetriever, load_chunks  # noqa: E402

SHOW = 5


def page_ranks(chunks: list[dict]) -> dict[tuple[str, int], int]:
    first: dict[tuple[str, int], int] = {}
    for c in chunks:
        first.setdefault((c["book_id"], c["page"]), c["rank"])
    return first


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
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
    lines = [f"# 사람이 분해한 질의로 단순 검색(재정렬 없음) — {spec['question_id']}", "", f"**원래 질문**: {spec['question']}", "",
             "방법: 질의 하나씩 **Dense(e5-large, 상위 100)** 와 **Hybrid(Dense 20 + BM25 20 → RRF 상위 40)** 로 검색. 재정렬·LLM 없음. 질의 문구는 사용자가 적은 그대로.", "",
             "## 요약: 정답 쪽이 몇 위에 나오는가 (같은 쪽의 여러 청크 중 가장 높은 순위, '밖'=후보 안에 없음)", "",
             "| 질의 | 질의 문구 | 정답 쪽 | Dense | Hybrid |", "|---|---|---|---|---|"]
    details = []
    for item in spec["queries"]:
        gold = [tuple(g) for g in item["gold_pages"]]
        d = dense.retrieve(item["query"], k=100).chunks
        h = hybrid.retrieve(item["query"], k=40).chunks
        dr, hr = page_ranks(d), page_ranks(h)
        fmt = lambda r, g: (f"{r[g]}위" if g in r else "밖")  # noqa: E731
        lines.append(f"| {item['id']} | {item['query']} | " + ", ".join(f"{b} p{p}" for b, p in gold) + " | " + ", ".join(fmt(dr, g) for g in gold) + " | " + ", ".join(fmt(hr, g) for g in gold) + " |")
        block = [f"### {item['id']}. {item['query']}", "", "정답 쪽: " + ", ".join(f"{b} p{p}" for b, p in gold), ""]
        for name, res in (("Dense", d), ("Hybrid", h)):
            block += [f"**{name} 상위 {SHOW}**", "", "| 순위 | 쪽 | 비고 |", "|---|---|---|"]
            for c in res[:SHOW]:
                block.append(f"| {c['rank']} | {c['book_id']} p{c['page']} | {'✓정답쪽' if (c['book_id'], c['page']) in gold else ''} |")
            block.append("")
        details += block
    lines += ["", "## 질의별 상위 5개", ""] + details
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[: 12 + len(spec["queries"])]))
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
