"""`.env` 로딩. 값이 비어 있는 변수는 설정하지 않은 것으로 취급한다.

`.env.example`을 복사하면 `HF_HOME=` 같은 빈 줄이 그대로 남는데, 빈 문자열은 "설정 없음"이 아니라
현재 폴더 기준 상대 경로로 해석되어 모델 캐시(수 GB)가 저장소 안(`./hub`)에 생긴다(STEP 3에서 실제로 발생).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    """프로젝트 루트의 `.env`를 읽고, 값이 비어 있는 변수는 환경에서 제거한다.

    huggingface_hub 등은 import 시점에 환경 변수를 읽으므로 이 함수를 그 라이브러리보다 먼저 호출해야 한다.
    """
    load_dotenv(ROOT / ".env")
    for name in [k for k, v in os.environ.items() if v.strip() == "" and k in _EMPTY_SENSITIVE]:
        del os.environ[name]


# 빈 값이면 동작이 달라지는 것으로 알려진 변수들
_EMPTY_SENSITIVE = {"HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "QDRANT_PATH", "QDRANT_URL", "QDRANT_API_KEY"}
