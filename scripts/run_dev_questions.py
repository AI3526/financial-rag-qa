"""STEP 9: development 30문항을 실제 API(Naive RAG)로 실행하고 기대 행동과 응답 상태를 비교한다.

사용: uv run python scripts/run_dev_questions.py --prompt answer_v1 --experiment-id dev-prompt-answer-v1-001

- 서비스에는 질문 문자열만 보낸다. 정답 파일(development.json)은 응답이 끝난 뒤 비교에만 읽는다.
- 로컬에만 저장한다(Langfuse 미사용, D-029). 결과 폴더가 이미 있으면 덮어쓰지 않고 중단한다.
- 결과: results/{experiment_id}/runs.jsonl(서비스가 기록), comparison.jsonl, summary.json, manifest.json
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
from src.evaluation.action_check import compare, request_body, summarize  # noqa: E402
from src.generation.prompts import load_prompt  # noqa: E402

EVAL_DIR = ROOT / "data" / "finance_basic" / "evaluation"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, help="prompts/{이름}.json (예: answer_v1)")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--rag-method", default="dense", choices=["dense", "hybrid", "hybrid_rerank"])
    args = parser.parse_args()
    out = ROOT / "results" / args.experiment_id
    if out.exists():
        print(f"이미 결과가 있습니다(덮어쓰지 않음): {out}. (중단)")
        return 1

    questions = json.loads((EVAL_DIR / "development_questions.json").read_text(encoding="utf-8"))  # id, question만
    app = build_default_app(prompt_name=args.prompt, use_langfuse=False)
    client = TestClient(app, raise_server_exceptions=False)
    headers = {"X-Experiment-Id": args.experiment_id, "X-Track": "development", "X-Run-Type": "general_eval"}
    responses = {}
    for q in questions:
        r = client.post("/rag/query", json=request_body(q, args.rag_method), headers=headers)
        responses[q["id"]] = (r.status_code, r.json())
        print(f"{q['id']}: HTTP {r.status_code} {r.json()['status']}")

    answers = {a["id"]: a for a in json.loads((EVAL_DIR / "development.json").read_text(encoding="utf-8"))}
    rows, review = [], []
    for q in questions:  # 여기서부터 정답 파일 사용(응답 이후)
        a = answers[q["id"]]
        http_status, resp = responses[q["id"]]
        row = compare(q["id"], a["expected_action"], resp, http_status)
        rows.append(row)
        review.append({**row, "question": q["question"], "answer": resp["answer"], "reason": resp["reason"],
                       "citations": resp["citations"], "reference_answer": a["reference_answer"],
                       "required_facts": a["required_facts"], "forbidden_claims": a["forbidden_claims"]})
    summary = summarize(rows)
    prompt = load_prompt(args.prompt)
    (out / "comparison.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in review) + "\n",
                                          encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "experiment_id": args.experiment_id, "track": "development", "step": "9", "question_set": "development",
        "questions": len(questions), "rag_method": args.rag_method, "prompt_version": prompt.version,
        "prompt_sha256": prompt.sha256, "langfuse": False,
        "note": "프롬프트 조정용 개발 점검. 답변 내용 정확성은 STEP 11에서 평가한다.",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n요약:", json.dumps(summary, ensure_ascii=False))
    print("불일치:", [(r["question_id"], r["expected_action"], r["status"]) for r in rows if not r["match"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
