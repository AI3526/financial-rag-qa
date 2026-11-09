"""STEP 7 준비: gpt-6-luna의 실제 응답 지연·토큰을 측정해 timeout 초기값의 근거로 쓴다 (DECISIONS.md D-030).

사용: uv run python scripts/measure_model_latency.py

- 입력은 실제 Dense 검색 문맥(Top-5) + 잠정 지시문이다. 최종 프롬프트가 아니다(STEP 8에서 확정).
- 질문은 평가 데이터가 아니라 직접 만든 것이다(정답·근거 원문 미사용).
- 로컬에만 저장한다(D-029: 튜닝·개발 실험은 Langfuse에 올리지 않음). 결과 폴더가 있으면 중단한다.
- 재시도 없이(max_retries=0) 호출해 원래 지연을 본다. 키 값은 출력하지 않는다.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_env  # noqa: E402

load_env()

from openai import OpenAI  # noqa: E402

from src.retrieval.dense import DenseRetriever  # noqa: E402

OUT = ROOT / "results" / "model-latency-001"
QUESTIONS = [
    ("근거 있는 질문", "2026년 상반기 영업이익은 얼마인가요?"),
    ("근거 부족 질문", "해온솔루션의 2027년 1분기 확정 매출을 알려 주세요."),
    ("모호한 질문", "리브의 재생 원료 비율은 얼마인가요?"),
]
EFFORTS = ["none", "low", "medium", "high"]
REPEATS = 2
SCHEMA = {
    "name": "rag_answer", "strict": True,
    "schema": {
        "type": "object", "additionalProperties": False,
        "required": ["status", "answer", "reason", "source_ids"],
        "properties": {
            "status": {"type": "string", "enum": ["answered", "needs_clarification", "abstained"]},
            "answer": {"type": "string"}, "reason": {"type": "string"},
            "source_ids": {"type": "array", "items": {"type": "string"}},
        },
    },
}
SYSTEM = ("제공된 자료만 근거로 한국어로 답한다. 자료에 없으면 abstained, 조건이 빠졌으면 needs_clarification으로 답하고, "
          "근거로 쓴 자료의 ID(S1, S2…)를 source_ids에 담는다.")


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    context = "\n\n".join(f"[S{i}] {c['text']}" for i, c in enumerate(chunks, start=1))
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"## 검색 자료\n{context}\n\n## 질문\n{question}"}]


def main() -> int:
    if OUT.exists():
        print(f"이미 결과가 있습니다(덮어쓰지 않음): {OUT}. (중단)")
        return 1
    model = os.environ["OPENAI_MODEL"]
    client = OpenAI(max_retries=0, timeout=180)
    retriever = DenseRetriever()
    rows = []
    for kind, question in QUESTIONS:
        messages = build_messages(question, retriever.retrieve(question).chunks)
        for effort in EFFORTS:
            for rep in range(1, REPEATS + 1):
                started = time.perf_counter()
                row = {"kind": kind, "effort": effort, "repeat": rep}
                try:
                    resp = client.chat.completions.create(
                        model=model, messages=messages, reasoning_effort=effort,
                        max_completion_tokens=4000,
                        response_format={"type": "json_schema", "json_schema": SCHEMA})
                    row["seconds"] = round(time.perf_counter() - started, 2)
                    usage = resp.usage
                    row.update(prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
                               reasoning_tokens=(usage.completion_tokens_details.reasoning_tokens
                                                 if usage.completion_tokens_details else None),
                               finish_reason=resp.choices[0].finish_reason)
                    try:
                        parsed = json.loads(resp.choices[0].message.content or "")
                        row["json_valid"], row["status"] = True, parsed.get("status")
                    except json.JSONDecodeError:
                        row["json_valid"], row["status"] = False, None
                except Exception as exc:  # noqa: BLE001 - 측정 목적, 오류 종류도 기록
                    row["seconds"] = round(time.perf_counter() - started, 2)
                    row["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                rows.append(row)
                print(f"{kind:8} effort={effort:6} #{rep}: {row['seconds']:6.2f}s "
                      f"in={row.get('prompt_tokens')} out={row.get('completion_tokens')} "
                      f"reason={row.get('reasoning_tokens')} status={row.get('status')} {row.get('error', '')}")

    OUT.mkdir(parents=True)
    with (OUT / "measurements.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {}
    for effort in EFFORTS:
        ok = [r["seconds"] for r in rows if r["effort"] == effort and "error" not in r]
        summary[effort] = {"n": len(ok), "mean": round(statistics.mean(ok), 2) if ok else None,
                           "max": max(ok) if ok else None,
                           "errors": sum(1 for r in rows if r["effort"] == effort and "error" in r)}
    (OUT / "manifest.json").write_text(json.dumps({
        "purpose": "timeout 초기값의 근거(지연 측정). 최종 프롬프트·설정이 아님", "model": model, "track": "development",
        "calls": len(rows), "max_completion_tokens": 4000, "retriever": "dense k=5", "summary_seconds": summary,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n요약(초):", json.dumps(summary, ensure_ascii=False))
    print(f"저장: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
