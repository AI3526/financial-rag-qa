"""확장 브랜치: 청킹 결과의 구조 품질을 측정한다 (모델 호출 없음, 평가 정답은 인용문 포함 여부 확인에만 사용).

사용: uv run python scripts/check_chunking_ext.py --dirs data/processed_intermediate data/processed_intermediate_v2a ...

지표(폴더마다):
- 청크 수, 글자 수 평균·중앙·최대, 30자 미만 청크 수(문장 꼬리).
- 근거 인용문이 같은 쪽의 한 청크에 온전히 들어 있는 비율(공백 제외). 제공 질문(development+additional)과 자체 질문(`evaluation_ext`)의 인용문을 쓴다.
- 표 머리줄 분리: 머리줄(열 이름·단위)과 숫자 행이 함께 있는 쪽 중에서, 머리줄이 숫자 행과 다른 청크에만 있는 쪽의 수(거친 규칙).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "finance_intermediate" / "evaluation"
EXT = ROOT / "data" / "finance_intermediate" / "evaluation_ext" / "ext_full.json"
HEADER_RE = re.compile(r"^항목 .*\(.*\)|\(백만원\)|\(억원\)|\(%\)")


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def load_quotes() -> list[tuple[str, str, int, str]]:
    out = []
    for f in ("development.json", "additional.json"):
        for q in json.loads((EVAL / f).read_text(encoding="utf-8")):
            for g in q["evidence_groups"]:
                out += [(q["id"], e["book_id"], e["page"], e["quote"]) for e in g]
    if EXT.exists():
        for q in json.loads(EXT.read_text(encoding="utf-8")):
            for g in q["evidence_groups"]:
                out += [(q["id"], e["book_id"], e["page"], e["quote"]) for e in g]
    return out


def measure(directory: Path, quotes) -> dict:
    chunks = [json.loads(line) for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    by_page: dict = defaultdict(list)
    for c in chunks:
        by_page[(c["book_id"], c["page"])].append(c["chunk_text"])
    sizes = [len(c["chunk_text"]) for c in chunks]
    whole = sum(1 for _, b, p, q in quotes if any(norm(q) in norm(t) for t in by_page[(b, p)]))
    tbl = sep = 0
    for texts in by_page.values():
        header_idx = {i for i, t in enumerate(texts) if any(HEADER_RE.search(ln) for ln in t.split("\n"))}
        row_idx = {i for i, t in enumerate(texts)
                   if sum(1 for ln in t.split("\n") if len(re.findall(r"\d[\d,\.]*", ln)) >= 3 and not HEADER_RE.search(ln)) >= 2}
        if header_idx and row_idx:
            tbl += 1
            sep += not (header_idx & row_idx)
    return {"dir": directory.name, "chunks": len(chunks), "avg_chars": round(statistics.mean(sizes), 1), "median_chars": statistics.median(sizes),
            "max_chars": max(sizes), "under_30_chars": sum(1 for x in sizes if x < 30), "quotes": len(quotes), "quotes_in_one_chunk": whole,
            "quotes_in_one_chunk_rate": round(whole / len(quotes), 4), "table_pages": tbl, "header_separated_pages": sep}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    args = ap.parse_args()
    quotes = load_quotes()
    for d in args.dirs:
        r = measure(ROOT / d, quotes)
        print(f"{r['dir']:34} 청크 {r['chunks']:5} 평균 {r['avg_chars']:6} 중앙 {r['median_chars']:5} 최대 {r['max_chars']:4} | 30자 미만 {r['under_30_chars']:3} | "
              f"근거 한 청크 {r['quotes_in_one_chunk']}/{r['quotes']} ({r['quotes_in_one_chunk_rate']:.1%}) | 표 머리줄 분리 {r['header_separated_pages']}/{r['table_pages']}쪽")
    return 0


if __name__ == "__main__":
    sys.exit(main())
