"""공식 비교 결과의 통계(순수 함수) (STEP 12, 12_ACCEPTANCE_CRITERIA §4.1·§4.2).

- 방식별 평균: 값이 있는 문항만 평균에 넣고 값이 없는 문항(해당 없음, 평가 불가)은 개수로 센다.
- 방식 쌍별 차이: 같은 질문끼리 짝지어 평균 차이와 **좋아진·나빠진·같은 문항 수**를 낸다. 95% 신뢰구간(부트스트랩)은 **참고용**이다.
- "비슷하다"의 정의: 평균 차이가 ±0.05 이내(설명을 위한 기준이며 통계적 동등성 검정이 아니다).
- 운영 지표: 일반 평가(`general_eval`) 요청만. 정상 건의 평균 응답시간, 오류율, 토큰, 확인 가능한 비용.
"""
from __future__ import annotations

from typing import Any

import numpy as np

SIMILAR = 0.05
TOL = 1e-9


def mean_or_none(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def group_stats(values: dict[Any, float | None]) -> dict:
    """키별 값(None 허용)의 평균. `n`은 값이 있는 문항 수, `missing`은 값이 없는 문항 수다."""
    vals = [v for v in values.values() if v is not None]
    return {"n": len(vals), "missing": len(values) - len(vals), "mean": (sum(vals) / len(vals)) if vals else None}


def bootstrap_ci(diffs: list[float], n_boot: int = 10000, seed: int = 20261110, alpha: float = 0.05) -> tuple[float, float] | None:
    """차이의 평균에 대한 부트스트랩 신뢰구간(참고용). 같은 seed면 같은 값이다."""
    if not diffs:
        return None
    arr = np.asarray(diffs, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(arr), size=(n_boot, len(arr)))
    means = np.sort(arr[idx].mean(axis=1))
    return float(means[int(alpha / 2 * n_boot)]), float(means[int((1 - alpha / 2) * n_boot) - 1])


def paired_stats(a: dict[Any, float | None], b: dict[Any, float | None], *, n_boot: int = 10000, seed: int = 20261110) -> dict:
    """같은 키(질문)에서 두 값이 모두 있는 것만 짝지어 `b - a`를 요약한다. `better`는 b가 a보다 좋아진(높은) 문항 수다."""
    keys = [k for k in a if k in b and a[k] is not None and b[k] is not None]
    diffs = [b[k] - a[k] for k in keys]
    n = len(diffs)
    mean_diff = (sum(diffs) / n) if n else None
    return {"n": n, "mean_a": (sum(a[k] for k in keys) / n) if n else None, "mean_b": (sum(b[k] for k in keys) / n) if n else None,
            "mean_diff": mean_diff, "better": sum(1 for d in diffs if d > TOL), "worse": sum(1 for d in diffs if d < -TOL),
            "same": sum(1 for d in diffs if abs(d) <= TOL), "ci95": bootstrap_ci(diffs, n_boot, seed),
            "similar": (abs(mean_diff) <= SIMILAR + TOL) if mean_diff is not None else None,
            "excluded": len(a) - n}


def operational(runs: list[dict]) -> dict:
    """한 방식의 일반 평가 요청들의 운영 지표. 사용량·비용을 확인하지 못한 요청은 0이 아니라 '확인 불가'로 세어 건수를 따로 적는다."""
    n = len(runs)
    errors = [r for r in runs if (r.get("response") or {}).get("status") == "error"]
    ok = [r for r in runs if (r.get("response") or {}).get("status") != "error"]

    def perf(r: dict, key: str) -> float | None:
        return (r.get("performance") or {}).get(key)

    def usage_known(r: dict) -> bool:
        calls = r.get("llm_calls") or []
        return bool(calls) and all(c.get("input_tokens") is not None and c.get("output_tokens") is not None for c in calls)

    known = [r for r in runs if usage_known(r)]
    costs = [r for r in runs if r.get("cost") is not None and r.get("cost_status") == "complete"]
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None  # noqa: E731
    return {
        "requests": n, "ok": len(ok), "errors": len(errors), "error_rate": (len(errors) / n) if n else None,
        "mean_total_ms": mean([perf(r, "total_duration_ms") for r in ok if perf(r, "total_duration_ms") is not None]),
        "mean_retrieval_ms": mean([perf(r, "retrieval_duration_ms") for r in ok if perf(r, "retrieval_duration_ms") is not None]),
        "mean_generation_ms": mean([perf(r, "generation_duration_ms") for r in ok if perf(r, "generation_duration_ms") is not None]),
        "input_tokens": sum(c["input_tokens"] for r in known for c in r["llm_calls"]),
        "output_tokens": sum(c["output_tokens"] for r in known for c in r["llm_calls"]),
        "usage_known_requests": len(known), "usage_unknown_requests": n - len(known),
        "cost_usd": sum(r["cost"] for r in costs), "cost_known_requests": len(costs), "cost_unknown_requests": n - len(costs),
        "retries": sum(max(0, len(r.get("llm_calls") or []) - 1) for r in runs),
    }
