"""확장 브랜치: 자체 추가 질문(abstain·clarify)을 자동으로 검증한다 (02_EVALUATION_PLAN §4.3, EXTENSION_INTERMEDIATE_LOG).

사용: uv run python scripts/verify_ext_questions.py --dataset intermediate

검증 항목(모두 자동, 사람 검증 아님):
- 스키마: 필수 필드, 허용된 expected_action(abstain·clarify), 고유한 ID, 제공 질문과 같은 ID·같은 문구가 없음.
- 근거 인용문: 질문이 지정한 (book_id, page)의 추출 텍스트에 공백을 제외하고 원문 그대로 존재.
- abstain(주제 부재형): `absent_terms`가 전체 쪽 텍스트(공백 제거)에 0건. abstain은 인용문 또는 부재 키워드 중 하나는 있어야 한다.
- clarify: 선택지가 2개 이상이고 서로 다르며, 근거 인용문이 2개 이상이거나 한 인용문에 서로 다른 선택지의 숫자 값이 2개 이상 들어 있어야 한다.
- 제공 질문과의 문자 2-gram 유사도 최댓값을 보고한다(0.6 이상이면 경고).
결과는 같은 폴더의 `verification_report.json`에 쓴다. 실패가 있으면 종료 코드 1.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"intermediate": {"processed": ROOT / "data" / "processed_intermediate", "eval": ROOT / "data" / "finance_intermediate" / "evaluation",
                             "ext": ROOT / "data" / "finance_intermediate" / "evaluation_ext"}}
REQUIRED = ("id", "split", "question", "reference_answer", "required_facts", "forbidden_claims", "expected_action", "evidence_groups",
            "evidence_required", "grading_notes", "conditions", "source", "verification")


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def bigrams(s: str) -> set[str]:
    s = norm(s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def verify(questions: list[dict], pages: dict, provided: list[dict]) -> list[dict]:
    full_text = {k: norm(v) for k, v in pages.items()}
    prov_ids = {p["id"] for p in provided}
    prov_bi = [(p["id"], bigrams(p["question"])) for p in provided]
    seen: set[str] = set()
    results = []
    for q in questions:
        problems: list[str] = []
        missing = [k for k in REQUIRED if k not in q]
        if missing:
            problems.append(f"필수 필드 없음: {missing}")
        if q.get("id") in seen or q.get("id") in prov_ids:
            problems.append("ID가 중복되거나 제공 질문의 ID와 같음")
        seen.add(q.get("id"))
        if q.get("expected_action") not in ("abstain", "clarify"):
            problems.append(f"허용되지 않은 expected_action: {q.get('expected_action')}")
        if any(p["question"].strip() == q.get("question", "").strip() for p in provided):
            problems.append("제공 질문과 문구가 같음")
        quotes_ok = quotes_total = 0
        for g in q.get("evidence_groups", []):
            for e in g:
                quotes_total += 1
                text = full_text.get((e["book_id"], e["page"]))
                if text is None:
                    problems.append(f"쪽 없음: {e['book_id']} p{e['page']}")
                elif norm(e["quote"]) in text:
                    quotes_ok += 1
                else:
                    problems.append(f"인용문이 쪽에 없음: {e['book_id']} p{e['page']} '{e['quote'][:30]}…'")
        terms = (q.get("verification") or {}).get("absent_terms") or []
        hits = {t: sum(1 for s in full_text.values() if norm(t) in s) for t in terms}
        for t, n in hits.items():
            if n:
                problems.append(f"부재 키워드 '{t}'가 {n}쪽에 존재")
        if q.get("expected_action") == "abstain" and not terms and quotes_total == 0:
            problems.append("abstain에 근거 인용문도 부재 키워드도 없음")
        if q.get("expected_action") == "clarify":
            opts = (q.get("ambiguity") or {}).get("options") or []
            if len(opts) < 2 or len(set(opts)) != len(opts):
                problems.append("clarify의 선택지가 2개 미만이거나 서로 같음")
            quote_text = norm("".join(e["quote"] for g in q.get("evidence_groups", []) for e in g))
            numbered = [[n for n in re.findall(r"\d[\d,\.]*", o)] for o in opts]
            covered = sum(1 for nums in numbered if nums and all(norm(n) in quote_text for n in nums))
            if quotes_total < 2 and covered < 2:  # 인용문이 2개 이상이거나, 서로 다른 선택지 값이 인용문에 2개 이상 들어 있어야 한다
                problems.append("clarify의 선택지 근거가 부족함(인용문 2개 미만이고 인용문에 든 선택지 값이 2개 미만)")
        qb = bigrams(q.get("question", ""))
        sim = max(((jaccard(qb, b), i) for i, b in prov_bi), default=(0.0, None))
        if sim[0] >= 0.6:
            problems.append(f"제공 질문 {sim[1]}과 유사도 {sim[0]:.2f}")
        results.append({"id": q.get("id"), "expected_action": q.get("expected_action"), "split": q.get("split"), "ok": not problems, "problems": problems,
                        "quotes": f"{quotes_ok}/{quotes_total}", "absent_term_hits": hits, "max_similarity_to_provided": {"id": sim[1], "jaccard": round(sim[0], 3)}})
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="intermediate", choices=sorted(DATASETS))
    args = ap.parse_args()
    d = DATASETS[args.dataset]
    questions = json.loads((d["ext"] / "ext_full.json").read_text(encoding="utf-8"))
    pages = {}
    for line in (d["processed"] / "pages.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = json.loads(line)
            pages[(p["book_id"], p["page"])] = p["text"]
    provided = []
    for f in ("development.json", "additional.json"):
        provided += json.loads((d["eval"] / f).read_text(encoding="utf-8"))
    results = verify(questions, pages, provided)
    report = {"dataset": args.dataset, "questions": len(results), "passed": sum(r["ok"] for r in results),
              "by_action": {a: sum(1 for r in results if r["expected_action"] == a) for a in ("abstain", "clarify")},
              "by_split": {s: sum(1 for r in results if r["split"] == s) for s in ("dev", "hold")}, "results": results,
              "note": "자동 검증이며 사람 검증이 아니다(verified_by_human=false)."}
    (d["ext"] / "verification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"질문 {report['questions']}개 중 통과 {report['passed']}개 {report['by_action']} {report['by_split']}")
    for r in results:
        flag = "OK " if r["ok"] else "실패"
        print(f"  {flag} {r['id']} [{r['expected_action']}/{r['split']}] 인용 {r['quotes']} 유사도 {r['max_similarity_to_provided']['jaccard']}" + (f" | {r['problems']}" if r["problems"] else ""))
    return 0 if report["passed"] == report["questions"] else 1


if __name__ == "__main__":
    sys.exit(main())
