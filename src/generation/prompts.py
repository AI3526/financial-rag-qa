"""프롬프트 파일 로딩 (STEP 8, 07_VERSIONING.md §7).

프롬프트의 원본은 Git의 `prompts/*.json` 파일이다(D-024). 파일의 sha256과 버전을 함께 기록해 어떤 프롬프트로
답했는지 추적한다. 기존 파일은 덮어쓰지 않고 바꿀 때는 새 버전 파일을 추가한다.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
DEFAULT_PROMPT = "answer_v2"  # STEP 9 조정 결과(D-034). 공식 실행 전에 동결한다


@dataclass(frozen=True)
class PromptSpec:
    version: str
    sha256: str
    system: str
    user: str


def load_prompt(name: str = DEFAULT_PROMPT, directory: Path = PROMPTS_DIR) -> PromptSpec:
    raw = (directory / f"{name}.json").read_bytes()
    data = json.loads(raw.decode("utf-8"))
    return PromptSpec(version=data["version"], sha256=hashlib.sha256(raw).hexdigest(),
                      system=data["system"], user=data["user"])
