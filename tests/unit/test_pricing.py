import json
import math

from src.generation import pricing as pr
from src.generation.pricing import call_cost, load_pricing, request_cost

P = load_pricing()


def call(inp, out, cached=0, written=0, tier="default"):
    return {"input_tokens": inp, "output_tokens": out, "cached_input_tokens": cached, "cache_write_tokens": written,
            "service_tier": tier}


def test_pricing_file_matches_the_values_provided_by_the_user():
    d = json.loads(pr.PRICING_PATH.read_text(encoding="utf-8"))
    assert (d["input"], d["cached_input"], d["cache_write"], d["output"]) == (0.10, 0.01, 0.125, 0.50)
    assert d["unit"] == "per 1M tokens" and d["currency"] == "USD"
    assert d["source"] == "https://developers.openai.com/api/docs/models/gpt-6-luna" and d["retrieved_at"]
    assert (P.input, P.cached_input, P.cache_write, P.output) == (0.10, 0.01, 0.125, 0.50)


def test_cost_without_cache():
    # 입력 100,000 + 출력 100,000 = (100,000×0.10 + 100,000×0.50) / 1M = 0.06달러
    assert math.isclose(call_cost(call(100_000, 100_000), P), 0.06)
    assert math.isclose(call_cost(call(1567, 75), P), (1567 * 0.10 + 75 * 0.50) / 1e6)


def test_cached_and_cache_write_tokens_use_their_own_rates():
    # 입력 200,000 중 캐시 읽기 80,000, 캐시 쓰기 20,000 → 일반 100,000
    expected = 100_000 * 0.10 + 80_000 * 0.01 + 20_000 * 0.125 + 0
    assert math.isclose(call_cost(call(200_000, 0, cached=80_000, written=20_000), P), expected / 1e6)
    # 입력 전체가 캐시에서 읽혔다면 일반 입력 비용은 0
    assert math.isclose(call_cost(call(1000, 0, cached=1000), P), 1000 * 0.01 / 1e6)


def test_reasoning_tokens_are_already_inside_output_and_not_added_again():
    c = {**call(1000, 200), "reasoning_tokens": 150}
    assert math.isclose(call_cost(c, P), (1000 * 0.10 + 200 * 0.50) / 1e6)


def test_unknown_values_give_none_not_zero():
    assert call_cost(call(None, 10), P) is None and call_cost(call(10, None), P) is None
    assert call_cost({**call(10, 10), "service_tier": None}, P) is None  # 처리 방식을 모르면 계산하지 않는다
    assert call_cost(call(10, 10, tier="flex"), P) is None  # 이 단가는 Standard 기준이다
    assert call_cost(call(100, 10, cached=80, written=80), P) is None  # 모순된 토큰 값
    assert call_cost(call(300_000, 10), P) is None  # 272K 초과 구간은 계산하지 않는다


def test_request_cost_sums_calls_and_marks_partial_when_a_call_is_unknown():
    r = request_cost([call(1000, 100), call(2000, 50)], P)
    assert r["cost_status"] == "complete" and math.isclose(r["cost"], (3000 * 0.10 + 150 * 0.50) / 1e6)
    assert r["currency"] == "USD" and "gpt-6-luna" in r["price_source"]
    partial = request_cost([call(1000, 100), call(None, None)], P)  # 예: timeout된 시도는 사용량을 모른다
    assert partial["cost_status"] == "partial" and math.isclose(partial["cost"], (1000 * 0.10 + 100 * 0.50) / 1e6)
    none = request_cost([call(None, None)], P)
    assert none == {"cost": None, "currency": None, "price_source": None, "cost_status": "unknown"}


def test_no_pricing_or_no_calls_means_no_cost():
    assert request_cost([call(1000, 100)], None)["cost"] is None
    assert request_cost([], P)["cost_status"] == "unknown"
