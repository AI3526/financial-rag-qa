"""STEP 9: 08 §9.4의 예외 질문 6종을 실제 API(Naive RAG)로 실행해 응답을 사람이 읽을 수 있게 저장한다.

사용: uv run python scripts/run_exception_questions.py --prompt answer_v1 --experiment-id exception-check-answer-v1-001

- 질문과 기대 동작은 **실행 전에** 이 파일에 고정했다. 질문은 평가 데이터가 아니라 직접 만든 것이다(정답 파일 미사용).
- `accepted_status`는 상태 점검용이고, 내용 기준(`content_check`)은 응답을 읽어 사람이 판단한다.
- 로컬에만 저장한다(Langfuse 미사용, D-029). 결과 폴더가 있으면 중단한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_env  # noqa: E402

load_env()

from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import build_default_app  # noqa: E402
from src.generation.prompts import load_prompt  # noqa: E402

CASES = [
    {"id": "EX-1", "kind": "필수 조건 누락", "question": "이번 분기 매출이 얼마인가요?",
     "accepted_status": ["needs_clarification"],
     "content_check": "회사와 기간(또는 항목)을 되묻고, 특정 수치를 임의로 고르지 않는다"},
    {"id": "EX-2", "kind": "서로 다른 회사 혼합", "question": "해온솔루션과 해온소재의 상반기 매출을 합치면 얼마인가요?",
     "accepted_status": ["answered", "abstained"],
     "content_check": "두 회사를 한 회사의 연결 실적처럼 합쳐 단정하지 않고, 회사별 수치와 별도 기준임을 구분한다"},
    {"id": "EX-3", "kind": "기간 불명확", "question": "작년 영업이익은 얼마였나요?",
     "accepted_status": ["needs_clarification", "abstained"],
     "content_check": "문서에 있는 2026년 상반기 수치를 '작년' 값으로 둔갑시키지 않는다"},
    {"id": "EX-4", "kind": "정정본과 이전 자료 충돌", "question": "해온솔루션의 8월 1일 사업개요에 나온 아시아 매출 값을 지금도 그대로 써도 되나요?",
     "accepted_status": ["answered"],
     "content_check": "정정 공지가 있으면 정정 후 값을 현재 값으로 안내하고 정정 전 값을 현재 값처럼 쓰지 않는다"},
    {"id": "EX-5", "kind": "검색 근거 없음", "question": "해온솔루션의 2027년 1분기 확정 매출을 알려 주세요.",
     "accepted_status": ["abstained"],
     "content_check": "검색한 자료에서 확인하지 못했다고 밝히고 수치를 만들지 않으며 세상에 없다고 단정하지 않는다"},
    {"id": "EX-6", "kind": "일부 근거만 확인", "question": "해온솔루션의 2026년 상반기 영업이익과 2027년 영업이익 전망을 알려 주세요.",
     "accepted_status": ["answered", "abstained"],
     "content_check": "확인되는 상반기 영업이익은 답하고, 2027년 전망은 확인되지 않았다고 구분해 밝힌다(부분 답변)"},
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    out = ROOT / "results" / args.experiment_id
    if out.exists():
        print(f"이미 결과가 있습니다(덮어쓰지 않음): {out}. (중단)")
        return 1
    client = TestClient(build_default_app(prompt_name=args.prompt, use_langfuse=False), raise_server_exceptions=False)
    headers = {"X-Experiment-Id": args.experiment_id, "X-Track": "development", "X-Run-Type": "general_eval"}
    rows = []
    for c in CASES:
        r = client.post("/rag/query", json={"question": c["question"], "rag_method": "dense", "question_id": c["id"]},
                        headers=headers)
        resp = r.json()
        rows.append({**c, "http_status": r.status_code, "status": resp["status"], "status_ok": resp["status"] in c["accepted_status"],
                     "answer": resp["answer"], "reason": resp["reason"], "citations": resp["citations"]})
        print(f"\n[{c['id']} {c['kind']}] {c['question']}\n  기대: {c['content_check']}\n  HTTP {r.status_code} | {resp['status']} "
              f"(허용 {c['accepted_status']}: {'O' if rows[-1]['status_ok'] else 'X'})\n  답변: {resp['answer']}\n  인용: "
              f"{[(x['file_name'], x['page']) for x in resp['citations']]}")
    prompt = load_prompt(args.prompt)
    (out).mkdir(parents=True, exist_ok=True)
    (out / "comparison.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "experiment_id": args.experiment_id, "track": "development", "step": "9", "purpose": "예외 질문 동작 확인(08 §9.4)",
        "prompt_version": prompt.version, "prompt_sha256": prompt.sha256, "langfuse": False,
        "status_ok": sum(r["status_ok"] for r in rows), "total": len(rows),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"\n상태 허용 범위 안: {sum(r['status_ok'] for r in rows)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
