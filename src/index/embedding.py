"""임베딩 후보 모델 정의와 인코딩 (STEP 3, 06_MODEL_SELECTION.md §11.2).

후보는 두 개로 제한한다: multilingual-e5-large, BGE-M3.
- e5는 문서에 `passage: `, 질문에 `query: ` 접두어를 붙이는 것이 모델 규칙이다.
- 벡터는 L2 정규화하고 거리는 코사인(= 정규화 벡터의 내적)으로 쓴다.
- 평가 정답·근거 원문은 이 모듈에 들어오지 않는다(입력은 청크 검색용 텍스트와 질문 문자열뿐).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EmbeddingSpec:
    key: str  # 짧은 이름 (컬렉션 이름에 쓰임)
    hf_id: str
    version_tag: str  # 컬렉션 이름용 버전 표기, 예: e5large-v1
    query_prefix: str
    passage_prefix: str
    max_seq_length: int


CANDIDATES: dict[str, EmbeddingSpec] = {
    "e5": EmbeddingSpec("e5", "intfloat/multilingual-e5-large", "e5large-v1", "query: ", "passage: ", 512),
    "bgem3": EmbeddingSpec("bgem3", "BAAI/bge-m3", "bgem3-v1", "", "", 1024),
}


def collection_name(dataset_id: str, chunking_version: str, spec: EmbeddingSpec) -> str:
    """`{dataset}__{chunking_version}__{embedding_version}` (07_VERSIONING.md)."""
    return f"{dataset_id}__{chunking_version}__{spec.version_tag}"


def cached_revision(hf_id: str) -> str | None:
    """로컬 HF 캐시에 받아 둔 모델 스냅샷의 revision(커밋 해시). 확인하지 못하면 None."""
    try:
        from huggingface_hub import snapshot_download

        # 캐시에 있는 스냅샷 경로의 마지막 폴더 이름이 커밋 해시다 (네트워크 호출 없음)
        return Path(snapshot_download(hf_id, local_files_only=True)).name
    except Exception:  # noqa: BLE001
        return None


class Embedder:
    """sentence-transformers 모델 래퍼. 모델 로드는 생성자에서 한 번만 한다."""

    def __init__(self, spec: EmbeddingSpec, device: str | None = None, batch_size: int = 16):
        import torch
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = SentenceTransformer(spec.hf_id, device=self.device)
        self.model.max_seq_length = spec.max_seq_length
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        self.revision = cached_revision(spec.hf_id)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self.spec.passage_prefix + t for t in texts])

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self.spec.query_prefix + t for t in texts])

    def token_lengths(self, texts: list[str]) -> list[int]:
        """접두어를 포함해 모델 토크나이저로 센 토큰 수(입력 한도 점검용)."""
        tok = self.model.tokenizer
        return [len(tok(self.spec.passage_prefix + t, add_special_tokens=True)["input_ids"]) for t in texts]
