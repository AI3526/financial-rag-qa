"""STEP 12: 공식 실행 전 사전 점검과 동결 파일, manifest (DECISIONS D-045, 07_VERSIONING §9.2).

사용:
    uv run python scripts/preflight_official.py                       # 점검만: 동결 파일과 현재 상태 대조 + 작업 트리 clean 확인
    uv run python scripts/preflight_official.py --allow-dirty         # 개발 중 점검(작업 트리 변경 허용, 동결 대조는 그대로)
    uv run python scripts/preflight_official.py --write-freeze        # 동결 파일 최초 생성(이미 있으면 중단, 덮어쓰지 않음)
    uv run python scripts/preflight_official.py --write-manifest official-001   # 점검 통과 시 results/official-001/manifest.json 작성

- 동결 파일(`configs/official_freeze.json`)은 사용자가 승인한 값(D-045)을 코드·데이터·설치 상태에서 읽어 만든 스냅샷이다. 커밋해 두고,
  공식 실행 전에 현재 상태와 한 글자도 다르지 않아야 한다. 다르면 중단한다(바꿨다면 새 실험 ID로 새로 동결).
- 공식 실행(`--write-manifest`)은 작업 트리가 clean일 때만 허용한다. manifest에 git commit, 동결 파일 해시, 실행 환경을 기록한다.
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

from src.evaluation.official import FREEZE_PATH, build_manifest, snapshot, verify_preflight  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-freeze", action="store_true")
    ap.add_argument("--write-manifest", metavar="EXPERIMENT_ID")
    ap.add_argument("--allow-dirty", action="store_true", help="작업 트리가 clean이 아니어도 점검만 진행(공식 manifest 작성에는 쓸 수 없음)")
    args = ap.parse_args()

    if args.write_freeze:
        if FREEZE_PATH.exists():
            print(f"동결 파일이 이미 있습니다(덮어쓰지 않음): {FREEZE_PATH}. (중단) 값을 바꿔야 하면 새 실험 ID로 새로 동결합니다.")
            return 1
        snap = snapshot()
        FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
        FREEZE_PATH.write_text(json.dumps(snap, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"동결 파일 작성: {FREEZE_PATH}")
        print("  이 파일을 커밋해야 공식 실행을 시작할 수 있습니다(작업 트리 clean 요건).")
        return 0

    if args.write_manifest and args.allow_dirty:
        print("공식 manifest는 작업 트리가 clean일 때만 작성합니다(--allow-dirty와 함께 쓸 수 없음). (중단)")
        return 1
    problems, current, git = verify_preflight(require_clean=not args.allow_dirty)
    print(f"git commit {git['commit'][:12]}, 변경 {len(git['status_lines'])}개")
    if problems:
        print(f"사전 점검 실패 ({len(problems)}건):")
        for p in problems:
            print("  -", p)
        return 1
    tail = "(작업 트리 clean 여부는 확인하지 않음)." if args.allow_dirty else "작업 트리도 clean입니다."
    print(f"사전 점검 통과: 동결 파일과 현재 상태가 같습니다. {tail}")
    if args.write_manifest:
        out = ROOT / "results" / args.write_manifest
        if (out / "manifest.json").exists() or (out / "runs.jsonl").exists():
            print(f"이미 결과가 있습니다(덮어쓰지 않음): {out}. (중단) 새 실험 ID를 쓰세요.")
            return 1
        out.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(args.write_manifest, current, git)
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"manifest 작성: {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
