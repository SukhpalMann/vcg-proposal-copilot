"""Session cost accounting (assignment deliverable 1c / 1d)."""
from __future__ import annotations

import config
from services import costing

USAGE = [
    {"stage": "extract", "provider": "ollama", "model": "gemma3:latest",
     "input_tokens": 3000, "output_tokens": 1200, "duration_seconds": 40.0},
    {"stage": "draft_sections", "provider": "ollama", "model": "gemma3:latest",
     "input_tokens": 2000, "output_tokens": 800, "duration_seconds": 25.0},
    {"stage": "draft_sections", "provider": "ollama", "model": "gemma3:latest",
     "input_tokens": 2000, "output_tokens": 800, "duration_seconds": 25.0},
]


def test_totals_aggregate_by_stage():
    t = costing.totals(USAGE)
    assert t["calls"] == 3
    assert t["input_tokens"] == 7000 and t["output_tokens"] == 2800
    assert t["by_stage"]["draft_sections"]["calls"] == 2
    assert t["seconds"] == 90.0


def test_local_cost_is_electricity_only_and_far_below_hosted():
    c = costing.session_cost(USAGE)
    # local marginal cost is electricity for the generating seconds
    expected_kwh = (config.LOCAL_DEVICE_WATTS * 90.0) / 3_600_000.0
    assert abs(c["local_inr"] - expected_kwh * config.ELECTRICITY_INR_PER_KWH) < 1e-6
    assert c["local_inr"] < c["api_equivalent_inr"]


def test_hosted_equivalent_uses_configured_rates():
    c = costing.session_cost(USAGE)
    expected_usd = (7000 / 1e6) * config.API_USD_PER_MTOK_INPUT + \
                   (2800 / 1e6) * config.API_USD_PER_MTOK_OUTPUT
    assert abs(c["api_equivalent_usd"] - expected_usd) < 1e-6
    assert abs(c["api_equivalent_inr"] - expected_usd * config.USD_INR) < 1e-3


def test_scale_projection_is_linear_in_sessions():
    one = costing.session_cost(USAGE)
    scaled = costing.at_scale(USAGE, users=10_000, sessions_per_user=1)
    assert scaled["sessions"] == 10_000
    assert scaled["input_tokens"] == one["input_tokens"] * 10_000
    assert abs(scaled["api_monthly_inr"] - one["api_equivalent_inr"] * 10_000) < 1.0
    assert scaled["self_hosted_gpu_hours"] > 0


def test_empty_usage_does_not_crash():
    c = costing.session_cost([])
    assert c["calls"] == 0 and c["input_tokens"] == 0 and c["local_inr"] == 0.0


def test_every_number_has_a_stated_assumption():
    keys = {k for k, _ in costing.assumptions()}
    assert {"Exchange rate", "Hosted input rate", "Hosted output rate",
            "Local device draw", "Electricity", "Rented GPU"} <= keys


def test_gpu_hours_come_from_tokens_not_laptop_seconds():
    scaled = costing.at_scale(USAGE, users=1000)
    expected = costing.gpu_seconds_for(7000 * 1000, 2800 * 1000) / 3600.0
    assert abs(scaled["self_hosted_gpu_hours"] - round(expected, 1)) < 0.05
    # laptop wall-clock (90 s/session) must not drive the GPU estimate
    assert scaled["self_hosted_gpu_hours"] != round(90.0 * 1000 / 3600.0, 1)


def test_self_hosted_bill_has_an_always_on_floor():
    scaled = costing.at_scale(USAGE, users=10)
    assert scaled["self_hosted_monthly_inr"] >= scaled["gpu_floor_inr"]


def test_every_hosted_tier_is_projected_and_recommendations_exist():
    scaled = costing.at_scale(USAGE)
    assert set(scaled["hosted_tiers_monthly_inr"]) == set(config.HOSTED_TIERS)
    recs = costing.recommendations(USAGE)
    assert recs and any("cache" in r.lower() for r in recs)
