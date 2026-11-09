"""run 레코드 저장 (STEP 7, 09_DATA_SCHEMA.md §9).

요청 1건 = JSONL 1행. 기존 내용은 지우지 않고 덧붙인다(append). 저장 실패는 응답에 영향을 주지 않는다.
- 실험 ID가 있으면 `{base}/{experiment_id}/runs.jsonl`, 없으면 `{base}/service/runs.jsonl`.
- 평가 전용 정보(정답, 근거 원문, expected_action)는 레코드에 넣지 않는다.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

EXPERIMENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")  # 경로로 쓰이므로 엄격하게 제한한다


class RunRecorder:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self._lock = threading.Lock()

    def path_for(self, experiment_id: str | None) -> Path:
        if experiment_id is not None and not EXPERIMENT_ID_RE.match(experiment_id):
            raise ValueError("올바르지 않은 experiment_id")
        return self.base_dir / (experiment_id or "service") / "runs.jsonl"

    def write(self, experiment_id: str | None, record: dict) -> bool:
        try:
            path = self.path_for(experiment_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False, default=str)
            with self._lock, path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("run 레코드 저장 실패: %s", type(exc).__name__)
            return False
