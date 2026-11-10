"""공식 비교 실험(STEP 12)의 동결 스냅샷, 사전 점검, 질문 목록, 결과 로더 (DECISIONS D-045).

- **동결 스냅샷**: 공식 실행에 영향을 주는 값(설정, 프롬프트 해시, 데이터 해시, 모델 revision, 패키지 버전, 평가 회차)을 한 곳에 모은다.
  `configs/official_freeze.json`에 저장해 커밋해 두고, 공식 실행 전에 현재 상태와 **한 글자도 다르지 않은지** 비교한다(다르면 중단).
- **사전 점검**: 작업 트리가 clean인지(`07_VERSIONING` §9.2), 동결 파일과 현재 상태가 같은지 확인한다.
- **질문 목록**: `*_questions.json`(질문 번호와 질문뿐)만 읽는다. 정답 파일은 평가 단계에서만 읽는다.
- **로더**: 서비스 기록(`runs.jsonl`)과 평가 결과를 (질문, 방식) 단위의 한 표로 합친다. 해당 없음과 평가 불가는 None이다(0으로 바꾸지 않음).
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import inspect
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from src.index.embedding import CANDIDATES, cached_revision

ROOT = Path(__file__).resolve().parents[2]
FREEZE_PATH = ROOT / "configs" / "official_freeze.json"
EVAL_DIR = ROOT / "data" / "finance_basic" / "evaluation"
PROCESSED = ROOT / "data" / "processed"
METHODS = ("dense", "hybrid", "hybrid_rerank")
SETS = (("development", "development_questions.json", "development.json"), ("additional", "additional_questions.json", "additional.json"))
PACKAGES = ("langchain-openai", "langchain-core", "langfuse", "qdrant-client", "rank-bm25", "sentence-transformers", "torch", "fastapi",
            "uvicorn", "httpx", "openai", "tiktoken", "pydantic", "numpy")
# 평가 설정(D-039, D-044, D-045). 평가 스크립트의 실제 설정은 각 결과 폴더의 manifest로 이 값과 대조한다(`verify_evaluation_manifest`).
EVALUATION = {"judge_prompt": "judge_v2", "judge_passes": 3, "judge_aggregation": "majority", "judge_max_output_tokens": 4096,
              "judge_reasoning_effort": "low", "faithfulness_prompt": "faithfulness_statement_v1", "faithfulness_passes": 3,
              "faithfulness_max_output_tokens": 4096, "ragas_version": "0.4.3", "retrieval_k": 5, "similar_threshold": 0.05}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ 질문 목록
def load_question_sets(eval_dir: Path = EVAL_DIR) -> list[dict]:
    """공식 실행 질문 60개(development 30 + additional 30). 질문 번호와 질문 문자열만 읽는다(정답 필드가 있으면 중단)."""
    out: list[dict] = []
    for set_type, questions_file, _ in SETS:
        for item in json.loads((eval_dir / questions_file).read_text(encoding="utf-8")):
            if set(item) != {"id", "question"}:
                raise ValueError(f"{questions_file}에 질문 번호와 질문 외의 필드가 있습니다: {sorted(item)}")
            out.append({"question_id": item["id"], "question": item["question"], "evaluation_set_type": set_type})
    ids = [q["question_id"] for q in out]
    if len(ids) != len(set(ids)):
        raise ValueError("질문 ID가 겹칩니다")
    return out


def question_set_of(eval_dir: Path = EVAL_DIR) -> dict[str, str]:
    return {q["question_id"]: q["evaluation_set_type"] for q in load_question_sets(eval_dir)}


# ------------------------------------------------------------------ 동결 스냅샷
def _packages() -> dict[str, str | None]:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def snapshot(*, revision_of: Callable[[str], str | None] = cached_revision) -> dict:
    """현재 코드·데이터·설치 상태에서 만든 동결 대상 스냅샷. `load_env()`가 먼저 호출되어 있어야 한다(`OPENAI_MODEL`)."""
    from rank_bm25 import BM25Okapi

    from src.api.pipeline import PipelineSettings
    from src.evaluation.faithfulness import load_statement_prompt
    from src.evaluation.judge import load_judge_prompt
    from src.generation.prompts import DEFAULT_PROMPT as ANSWER_DEFAULT, load_prompt
    from src.retrieval.dense import RetrievalConfig
    from src.retrieval.hybrid import HybridConfig
    from src.retrieval.rerank import RERANKER_ID, RerankConfig

    answer = load_prompt(ANSWER_DEFAULT)
    judge = load_judge_prompt(EVALUATION["judge_prompt"])
    faith = load_statement_prompt()
    rc = RetrievalConfig()
    bm25 = inspect.signature(BM25Okapi.__init__).parameters
    docs = [json.loads(l) for l in (PROCESSED / "documents.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    qsets = load_question_sets()
    snap = {
        "schema": 1,
        "pipeline_settings": asdict(PipelineSettings()),
        "answer_prompt": {"name": ANSWER_DEFAULT, "version": answer.version, "sha256": answer.sha256},
        "judge_prompt": {"name": EVALUATION["judge_prompt"], "version": judge["version"], "sha256": judge["sha256"]},
        "faithfulness_prompt": {"name": EVALUATION["faithfulness_prompt"], "version": faith["version"], "sha256": faith["sha256"]},
        "retrieval": {"dense": {"dataset_id": rc.dataset_id, "dataset_version": rc.dataset_version, "chunking_version": rc.chunking_version,
                                "embedding": rc.embedding, "embedding_version": rc.spec.version_tag, "collection": rc.collection, "k": rc.k},
                      "hybrid": asdict(HybridConfig()), "rerank": asdict(RerankConfig()),
                      "bm25": {"tokenizer": "whitespace", "k1": bm25["k1"].default, "b": bm25["b"].default, "epsilon": bm25["epsilon"].default}},
        "models": {"generation": os.environ.get("OPENAI_MODEL"),
                   "embedding": {"hf_id": rc.spec.hf_id, "revision": revision_of(rc.spec.hf_id)},
                   "reranker": {"id": RERANKER_ID, "revision": revision_of(RERANKER_ID)}},
        "evaluation": dict(EVALUATION),
        "data": {"documents_sha256": sha256_file(PROCESSED / "documents.jsonl"), "pages_sha256": sha256_file(PROCESSED / "pages.jsonl"),
                 "chunks_sha256": sha256_file(PROCESSED / "chunks.jsonl"),
                 "pdfs": {d["file_name"]: d["file_sha256"] for d in docs},
                 "evaluation_files": {name: sha256_file(EVAL_DIR / name) for _, q, a in SETS for name in (q, a)},
                 "question_ids": [q["question_id"] for q in qsets]},
        "docs": {"12_ACCEPTANCE_CRITERIA.md": sha256_file(ROOT / "project_docs" / "01_core" / "12_ACCEPTANCE_CRITERIA.md")},
        "packages": _packages(), "python": sys.version.split()[0],
    }
    # JSON으로 저장했다가 다시 읽은 값과 비교하므로 처음부터 JSON 왕복으로 정규화한다(튜플이 리스트로 바뀌어 생기는 가짜 차이 방지)
    return json.loads(json.dumps(snap, ensure_ascii=False))


def diff_snapshots(frozen: Any, current: Any, path: str = "") -> list[str]:
    """두 스냅샷의 차이를 `경로: 동결값 → 현재값` 문자열 목록으로 돌려준다. 같으면 빈 목록."""
    if isinstance(frozen, dict) and isinstance(current, dict):
        out: list[str] = []
        for k in sorted(set(frozen) | set(current)):
            sub = f"{path}.{k}" if path else str(k)
            if k not in frozen:
                out.append(f"{sub}: (동결에 없음) → {current[k]!r}")
            elif k not in current:
                out.append(f"{sub}: {frozen[k]!r} → (현재 없음)")
            else:
                out += diff_snapshots(frozen[k], current[k], sub)
        return out
    return [] if frozen == current else [f"{path}: {frozen!r} → {current!r}"]


# ------------------------------------------------------------------ git 상태
def parse_porcelain(text: str) -> list[str]:
    return [l for l in text.splitlines() if l.strip()]


def git_state(cwd: Path = ROOT) -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8", check=True).stdout

    lines = parse_porcelain(run("status", "--porcelain"))
    return {"commit": run("rev-parse", "HEAD").strip(), "dirty": bool(lines), "status_lines": lines}


def verify_preflight(*, require_clean: bool = True, freeze_path: Path = FREEZE_PATH) -> tuple[list[str], dict, dict]:
    """(문제 목록, 현재 스냅샷, git 상태). 문제 목록이 비어 있어야 공식 실행을 시작할 수 있다."""
    problems: list[str] = []
    git = git_state()
    if require_clean and git["dirty"]:
        problems.append(f"작업 트리가 clean이 아닙니다({len(git['status_lines'])}개 변경): {git['status_lines'][:5]}")
    if not freeze_path.exists():
        problems.append(f"동결 파일이 없습니다: {freeze_path}")
        return problems, {}, git
    current = snapshot()
    problems += [f"동결과 다름: {d}" for d in diff_snapshots(json.loads(freeze_path.read_text(encoding="utf-8")), current)]
    return problems, current, git


def build_manifest(experiment_id: str, snap: dict, git: dict, *, freeze_path: Path = FREEZE_PATH, extra: dict | None = None) -> dict:
    """실험 설정 스냅샷(`07_VERSIONING` §9.2). 같은 값이 동결 파일에도 있고, 여기에는 git 상태와 실행 환경이 더해진다."""
    from datetime import datetime, timezone

    gpu = None
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        pass
    return {"experiment_id": experiment_id, "track": "official", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git["commit"], "git_dirty": git["dirty"], "freeze_file": str(freeze_path.relative_to(ROOT)).replace("\\", "/"),
            "freeze_sha256": sha256_file(freeze_path), "snapshot": snap,
            "environment": {"platform": platform.platform(), "python": sys.version.split()[0], "gpu": gpu},
            "plan": {"questions": len(snap["data"]["question_ids"]), "methods": list(METHODS),
                     "answers_expected": len(snap["data"]["question_ids"]) * len(METHODS)}, **(extra or {})}


# ------------------------------------------------------------------ 실행 기록이 동결과 맞는가
def check_runs_against_freeze(records: list[dict], frozen: dict) -> list[str]:
    """서버가 남긴 run 레코드의 버전 필드가 동결 값과 같은지, 방식마다 `config_version`이 하나뿐인지 확인한다."""
    problems: list[str] = []
    expect = {"prompt_sha256": frozen["answer_prompt"]["sha256"], "prompt_version": frozen["answer_prompt"]["version"],
              "embedding_version": frozen["retrieval"]["dense"]["embedding_version"], "chunking_version": frozen["retrieval"]["dense"]["chunking_version"],
              "dataset_version": frozen["retrieval"]["dense"]["dataset_version"], "model_id": frozen["models"]["generation"]}
    config_versions: dict[str, set] = {}
    for r in records:
        v = r.get("versions") or {}
        for k, want in expect.items():
            if v.get(k) != want:
                problems.append(f"{r.get('question_id')}/{r.get('rag_method')}: versions.{k}={v.get(k)!r}, 동결 {want!r}")
        config_versions.setdefault(r.get("rag_method"), set()).add(v.get("config_version"))
    for m, s in config_versions.items():
        if len(s) > 1:
            problems.append(f"{m}: config_version이 여러 개입니다 {sorted(map(str, s))}")
    return problems


def verify_evaluation_manifest(kind: str, manifest: dict, frozen: dict) -> list[str]:
    """평가 결과 폴더의 manifest가 동결된 평가 설정과 같은지 확인한다(kind: judge | faithfulness)."""
    ev = frozen["evaluation"]
    problems: list[str] = []
    if kind == "judge":
        pr = manifest.get("prompt") or {}
        if pr.get("sha256") != frozen["judge_prompt"]["sha256"]:
            problems.append("judge 프롬프트 sha256이 동결과 다름")
        if manifest.get("passes") != ev["judge_passes"]:
            problems.append(f"judge 회차 {manifest.get('passes')} ≠ 동결 {ev['judge_passes']}")
        if manifest.get("max_output_tokens") != ev["judge_max_output_tokens"]:
            problems.append("judge 출력 한도가 동결과 다름")
    elif kind == "faithfulness":
        if (manifest.get("statement_prompt") or {}).get("sha256") != frozen["faithfulness_prompt"]["sha256"]:
            problems.append("faithfulness 프롬프트 sha256이 동결과 다름")
        if manifest.get("passes") != ev["faithfulness_passes"]:
            problems.append(f"faithfulness 회차 {manifest.get('passes')} ≠ 동결 {ev['faithfulness_passes']}")
        if manifest.get("max_output_tokens") != ev["faithfulness_max_output_tokens"]:
            problems.append("faithfulness 출력 한도가 동결과 다름")
        if (manifest.get("tools") or {}).get("ragas") != ev["ragas_version"]:
            problems.append(f"ragas 버전 {(manifest.get('tools') or {}).get('ragas')} ≠ 동결 {ev['ragas_version']}")
    else:
        problems.append(f"알 수 없는 평가 종류: {kind}")
    if manifest.get("model") != frozen["models"]["generation"]:
        problems.append("평가 모델이 동결과 다름")
    return problems


# ------------------------------------------------------------------ 결과 로더
def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_runs(sources: list[str], base: Path = ROOT / "results") -> dict[tuple[str, str], dict]:
    """(질문 ID, 방식) → run 레코드. 일반 평가(`general_eval`)만 담고, 같은 키가 두 번 있으면 중단한다(덮어쓰지 않음)."""
    out: dict[tuple[str, str], dict] = {}
    for s in sources:
        for r in _read_jsonl(base / s / "runs.jsonl"):
            if r.get("run_type") != "general_eval":
                continue
            key = (r["question_id"], r["rag_method"])
            if key in out:
                raise ValueError(f"같은 (질문, 방식)이 두 번 있습니다: {key}")
            out[key] = r
    return out


def load_retrieval(folders: list[str], base: Path = ROOT / "results") -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for f in folders:
        for r in _read_jsonl(base / f / "retrieval_eval.jsonl"):
            out[(r["question_id"], r["rag_method"])] = r["metrics"]
    return out


def faith_per_request(rows: list[dict]) -> dict[str, dict]:
    """요청별 Faithfulness. 점수가 있는 회차의 평균(회차 사이 변동은 `passes`에 남김). 점수가 하나도 없으면 평가 불가(있으면) 또는 해당 없음."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["request_id"], []).append(r)
    out = {}
    for rid, rs in by.items():
        scored = [r["score"] for r in rs if r["status"] == "scored"]
        status = "scored" if scored else ("unevaluable" if any(r["status"] == "unevaluable" for r in rs) else "not_applicable")
        out[rid] = {"status": status, "score": (sum(scored) / len(scored)) if scored else None,
                    "passes": [r["score"] if r["status"] == "scored" else r["status"] for r in sorted(rs, key=lambda r: r.get("pass", 1))]}
    return out


def load_faithfulness(folders: list[str], base: Path = ROOT / "results") -> dict[str, dict]:
    rows: list[dict] = []
    for f in folders:
        rows += _read_jsonl(base / f / "faithfulness.jsonl")
    return faith_per_request(rows)


def load_judge(folders: list[str], base: Path = ROOT / "results") -> dict[str, dict]:
    """요청 ID → Judge 행. 여러 회차 폴더는 `judge_majority.jsonl`(다수결), 1회 채점 폴더는 `judge.jsonl`을 그대로 쓴다."""
    out: dict[str, dict] = {}
    for f in folders:
        p = base / f / ("judge_majority.jsonl" if (base / f / "judge_majority.jsonl").exists() else "judge.jsonl")
        for r in _read_jsonl(p):
            out[r["request_id"]] = r
    return out


JUDGE_ITEMS = ("facts", "conditions", "evidence", "decision")
METRICS = (("status_match", "기대 동작과 상태 일치"), ("hit", "Hit@5"), ("recall", "Recall@5"), ("faith", "Faithfulness"),
           ("j_facts", "Judge 사실"), ("j_conditions", "Judge 조건"), ("j_evidence", "Judge 근거"), ("j_decision", "Judge 판단"))


def build_table(runs: dict, refs: dict, retrieval: dict, faith: dict, judge: dict, set_of: dict) -> dict[tuple[str, str], dict]:
    """(질문 ID, 방식) → 한 행. 값이 없는 지표(해당 없음, 평가 불가, 평가 결과 없음)는 None이다."""
    from src.evaluation.judge import expected_status

    table: dict[tuple[str, str], dict] = {}
    for (qid, method), rec in runs.items():
        ref = refs[qid]
        status = (rec.get("response") or {}).get("status")
        ret = retrieval.get((qid, method)) or {}
        fa = faith.get(rec["request_id"]) or {}
        jd = (judge.get(rec["request_id"]) or {}).get("items") or {}
        row = {"question_id": qid, "rag_method": method, "request_id": rec["request_id"], "trace_id": rec.get("trace_id"), "evaluation_set_type": set_of.get(qid),
               "expected_action": ref["expected_action"], "status": status,
               "status_match": 1 if status == expected_status(ref["expected_action"]) else 0,
               "hit": ret.get("hit") if ret.get("status") == "scored" else None,
               "recall": ret.get("recall") if ret.get("status") == "scored" else None,
               "faith": fa.get("score") if fa.get("status") == "scored" else None}
        for k in JUDGE_ITEMS:
            it = jd.get(k) or {}
            row[f"j_{k}"] = it.get("score") if it.get("status") == "scored" else None
        row["faith_status"] = fa.get("status")
        row["judge_status"] = {k: (jd.get(k) or {}).get("status") for k in JUDGE_ITEMS}
        table[(qid, method)] = row
    return table
