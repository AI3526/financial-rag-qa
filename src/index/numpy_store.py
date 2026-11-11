"""numpy 정확 검색 저장소 (확장 브랜치 extension/intermediate, DECISIONS D-047).

Qdrant 없이 임베딩 행렬 하나와 청크 메타데이터로 코사인 Top-k를 구한다. 벡터는 L2 정규화되어 있으므로 내적이 곧 코사인 유사도다.
- 정확 검색(근사 인덱스 없음)이라 검색 알고리즘 비교에 정확도 차이가 섞이지 않는다.
- 저장 형식(폴더 하나): `vectors.npy`(float32, n×d), `payloads.jsonl`(`qdrant_store.chunk_payload`와 같은 필드), `meta.json`.
- 이미 있는 폴더는 덮어쓰지 않는다(CLAUDE.md §4). 평가 정답·근거는 저장하지 않는다.
- 같은 점수는 청크 번호가 작은 쪽이 먼저 나오도록 안정 정렬해 결과가 항상 같다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.index.qdrant_store import chunk_payload

ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT / "data" / "numpy_index"


@dataclass
class NumpyIndex:
    vectors: np.ndarray  # (n, d) float32, 정규화
    payloads: list[dict]
    meta: dict

    def __post_init__(self):
        if self.vectors.ndim != 2 or len(self.payloads) != self.vectors.shape[0]:
            raise ValueError("벡터 행 수와 메타데이터 수가 같아야 합니다.")

    def __len__(self) -> int:
        return len(self.payloads)

    def search(self, vector, k: int = 5) -> list[dict]:
        """상위 k개를 `{score, **payload}`로 돌려준다(`qdrant_store.search`와 같은 형식)."""
        q = np.asarray(vector, dtype=np.float32)
        if q.shape != (self.vectors.shape[1],):
            raise ValueError(f"질문 벡터 차원이 다릅니다: {q.shape} ≠ ({self.vectors.shape[1]},)")
        scores = self.vectors @ q
        order = np.argsort(-scores, kind="stable")[:k]
        return [{"score": float(scores[i]), **self.payloads[i]} for i in order]

    @classmethod
    def build(cls, chunks: list[dict], vectors, meta: dict) -> "NumpyIndex":
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1)
        if arr.shape[0] and not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("벡터가 정규화되어 있지 않습니다(내적이 코사인이 되지 않음).")
        return cls(arr, [chunk_payload(c) for c in chunks], {**meta, "count": len(chunks), "dimension": int(arr.shape[1])})

    def save(self, name: str, base: Path = INDEX_DIR) -> Path:
        out = base / name
        if out.exists():
            raise FileExistsError(f"이미 있는 색인은 덮어쓰지 않습니다: {out}")
        out.mkdir(parents=True)
        np.save(out / "vectors.npy", self.vectors)
        with (out / "payloads.jsonl").open("w", encoding="utf-8") as fh:
            for p in self.payloads:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
        (out / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return out

    @classmethod
    def load(cls, name: str, base: Path = INDEX_DIR) -> "NumpyIndex":
        d = base / name
        payloads = [json.loads(line) for line in (d / "payloads.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        return cls(np.load(d / "vectors.npy"), payloads, json.loads((d / "meta.json").read_text(encoding="utf-8")))
