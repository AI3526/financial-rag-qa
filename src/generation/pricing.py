"""모델 호출 비용 계산 (STEP 8, 09_DATA_SCHEMA.md §9 "비용은 확인 가능한 단가가 있을 때만").

단가 원본은 Git의 `configs/pricing.json`이다(D-033). 이 단가는 Standard 처리 기준이다. 응답의 `service_tier`가
`applies_when_service_tier`와 다르거나, 계산에 필요한 토큰 값이 없으면 그 호출의 비용은 계산하지 않고 null로 둔다.
확인하지 못한 값을 0으로 바꾸지 않는다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PRICING_PATH = Path(__file__).resolve().parents[2] / "configs" / "pricing.json"
_MILLION = 1_000_000


@dataclass(frozen=True)
class Pricing:
    model: str
    currency: str
    input: float
    cached_input: float
    cache_write: float
    output: float
    applies_when_service_tier: str
    source: str
    retrieved_at: str


def load_pricing(path: Path = PRICING_PATH) -> Pricing:
    d = json.loads(path.read_text(encoding="utf-8"))
    return Pricing(model=d["model"], currency=d["currency"], input=d["input"], cached_input=d["cached_input"],
                   cache_write=d["cache_write"], output=d["output"],
                   applies_when_service_tier=d["applies_when_service_tier"], source=d["source"],
                   retrieved_at=d["retrieved_at"])


def call_cost(call: dict, p: Pricing) -> float | None:
    """llm_calls의 한 행의 비용. 계산할 수 없으면 None."""
    inp, out = call.get("input_tokens"), call.get("output_tokens")
    if inp is None or out is None or call.get("service_tier") != p.applies_when_service_tier:
        return None
    cached = call.get("cached_input_tokens") or 0
    written = call.get("cache_write_tokens") or 0
    uncached = inp - cached - written
    if uncached < 0 or inp > 272_000:  # 272K 초과 구간은 단가가 달라 이 파일로 계산하지 않는다
        return None
    return (uncached * p.input + cached * p.cached_input + written * p.cache_write + out * p.output) / _MILLION


def request_cost(calls: list[dict], p: Pricing | None) -> dict:
    """요청 1건의 비용 필드. 일부 호출의 비용을 모르면 status=partial(알려진 합계는 하한)로 표시한다."""
    if p is None or not calls:
        return {"cost": None, "currency": None, "price_source": None, "cost_status": "unknown"}
    costs = [call_cost(c, p) for c in calls]
    known = [c for c in costs if c is not None]
    status = "complete" if len(known) == len(costs) else ("partial" if known else "unknown")
    return {"cost": round(sum(known), 8) if known else None, "currency": p.currency if known else None,
            "price_source": f"{p.source} (retrieved {p.retrieved_at})" if known else None, "cost_status": status}
