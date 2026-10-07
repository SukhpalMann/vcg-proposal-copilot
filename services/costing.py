"""Session cost accounting for the generation stages.

The assignment requires a measured per-session cost in tokens and rupees, plus a
projection to 10,000 users. This module turns the per-call records in
``state["model_usage"]`` into both, and -- because the demo runs a LOCAL model --
reports the local cost and the hosted-API equivalent side by side so the
quality / speed / cost trade-off is explicit rather than asserted.

Every rate below is a configurable assumption, not a measurement. Set them from
live pricing before quoting a figure; `assumptions()` prints exactly what was
used so a number can never be presented without its basis.
"""
from __future__ import annotations

import config


def _tok(event: dict, key: str) -> int:
    return int(event.get(key) or 0)


def totals(model_usage: list[dict]) -> dict:
    """Aggregate the raw per-call usage records."""
    calls = list(model_usage or [])
    by_stage: dict[str, dict] = {}
    for e in calls:
        s = by_stage.setdefault(e.get("stage") or "unknown",
                                {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                 "seconds": 0.0})
        s["calls"] += 1
        s["input_tokens"] += _tok(e, "input_tokens")
        s["output_tokens"] += _tok(e, "output_tokens")
        s["seconds"] += float(e.get("duration_seconds") or 0.0)
    return {
        "calls": len(calls),
        "input_tokens": sum(_tok(e, "input_tokens") for e in calls),
        "output_tokens": sum(_tok(e, "output_tokens") for e in calls),
        "seconds": round(sum(float(e.get("duration_seconds") or 0.0) for e in calls), 2),
        "by_stage": by_stage,
        "provider": (calls[0].get("provider") if calls else None),
        "model": (calls[0].get("model") if calls else None),
    }


def session_cost(model_usage: list[dict]) -> dict:
    """Per-session cost, local and hosted-API equivalent, in INR."""
    t = totals(model_usage)

    # Local: no per-token charge. The marginal cost is electricity for the
    # seconds the GPU/NPU was busy.
    kwh = (config.LOCAL_DEVICE_WATTS * t["seconds"]) / 3_600_000.0
    local_inr = kwh * config.ELECTRICITY_INR_PER_KWH

    # Hosted equivalent, at the configured per-million-token rates.
    api_usd = (t["input_tokens"] / 1e6) * config.API_USD_PER_MTOK_INPUT + \
              (t["output_tokens"] / 1e6) * config.API_USD_PER_MTOK_OUTPUT
    api_inr = api_usd * config.USD_INR

    return {
        **t,
        "local_inr": round(local_inr, 4),
        "api_equivalent_usd": round(api_usd, 6),
        "api_equivalent_inr": round(api_inr, 4),
    }


def _hosted_usd(input_tokens: int, output_tokens: int, rates: tuple[float, float]) -> float:
    return (input_tokens / 1e6) * rates[0] + (output_tokens / 1e6) * rates[1]


def gpu_seconds_for(input_tokens: int, output_tokens: int) -> float:
    """GPU time for one session, from its TOKENS and a declared GPU throughput.

    The earlier projection multiplied the laptop's wall-clock seconds by the
    session count, which priced a rented GPU as though it were an M4 laptop
    running on CPU-class throughput. GPU time is derived from tokens instead.
    """
    return (input_tokens / config.GPU_PREFILL_TOK_PER_SEC
            + output_tokens / config.GPU_OUTPUT_TOK_PER_SEC)


def at_scale(model_usage: list[dict], users: int = 10_000,
             sessions_per_user: int = 1) -> dict:
    """Monthly projection. One session = one proposal drafted.

    Reports every hosted tier side by side and two self-hosted figures: the GPU
    hours the work actually needs, and the realistic bill, which has a floor of
    one GPU rented around the clock (you cannot rent 3 minutes of a GPU at a time
    and still answer a user who arrives at 2am)."""
    c = session_cost(model_usage)
    sessions = users * sessions_per_user
    tin, tout = c["input_tokens"] * sessions, c["output_tokens"] * sessions

    hosted_tiers = {
        name: round(_hosted_usd(tin, tout, rates) * config.USD_INR, 2)
        for name, rates in config.HOSTED_TIERS.items()
    }
    gpu_hours = gpu_seconds_for(tin, tout) / 3600.0
    busy_inr = gpu_hours * config.GPU_INR_PER_HOUR
    floor_inr = config.GPU_HOURS_PER_MONTH * config.GPU_INR_PER_HOUR
    self_hosted_inr = max(busy_inr, floor_inr)

    per_session_api = c["api_equivalent_inr"]
    break_even = int(floor_inr / per_session_api) if per_session_api > 0 else None

    return {
        "users": users,
        "sessions_per_user": sessions_per_user,
        "sessions": sessions,
        "input_tokens": tin,
        "output_tokens": tout,
        "api_monthly_inr": round(_hosted_usd(tin, tout, (config.API_USD_PER_MTOK_INPUT,
                                                          config.API_USD_PER_MTOK_OUTPUT))
                                 * config.USD_INR, 2),
        "hosted_tiers_monthly_inr": hosted_tiers,
        "self_hosted_gpu_hours": round(gpu_hours, 1),
        "self_hosted_busy_inr": round(busy_inr, 2),
        "self_hosted_monthly_inr": round(self_hosted_inr, 2),
        "gpu_floor_inr": round(floor_inr, 2),
        "break_even_sessions": break_even,
        "input_share": (round(c["input_tokens"] / (c["input_tokens"] + c["output_tokens"]), 2)
                        if (c["input_tokens"] + c["output_tokens"]) else 0.0),
    }


def recommendations(model_usage: list[dict], users: int = 10_000) -> list[str]:
    """What we would change at scale, derived from the numbers rather than asserted."""
    s = at_scale(model_usage, users=users)
    out: list[str] = []
    if s["sessions"] == 0 or s["input_tokens"] == 0:
        return out
    be = s["break_even_sessions"]
    if s["api_monthly_inr"] < s["self_hosted_monthly_inr"]:
        out.append(
            f"Serve the hosted tier by default: ₹{s['api_monthly_inr']:,.0f}/month on "
            f"{config.HOSTED_MODEL_LABEL} against a ₹{s['gpu_floor_inr']:,.0f} floor for one "
            f"always-on GPU. Self-hosting only pays past about {be:,} proposals a month."
            if be else "Serve the hosted tier by default.")
        out.append("Keep the local Ollama mode as a premium option for firms whose "
                   "tenders may not leave their network; there the GPU floor is the "
                   "price of the guarantee.")
    else:
        out.append(
            f"At this volume one dedicated GPU (₹{s['self_hosted_monthly_inr']:,.0f}/month) "
            f"beats the hosted API (₹{s['api_monthly_inr']:,.0f}); move to self-hosting.")
    if s["input_share"] >= 0.6:
        out.append(
            f"{s['input_share']:.0%} of tokens are input (tender plus evidence passages, "
            f"resent for every section). Cache the shared prefix: cached reads are billed "
            f"at a fraction of the input rate.")
    out.append("Skip the model call for sections that will be an EVIDENCE GAP anyway; "
               "the verifier would block that text regardless.")
    out.append("Keep verification rule-based: it costs no tokens, so cost grows only "
               "with drafting.")
    return out


def assumptions() -> list[tuple[str, str]]:
    """Every rate behind the numbers above, so a figure is never shown bare."""
    rows = [
        ("Exchange rate", f"₹{config.USD_INR:.0f} per USD"),
        ("Hosted model", config.HOSTED_MODEL_LABEL),
        ("Hosted input rate", f"${config.API_USD_PER_MTOK_INPUT}/M tokens"),
        ("Hosted output rate", f"${config.API_USD_PER_MTOK_OUTPUT}/M tokens"),
    ]
    for name, (i, o) in config.HOSTED_TIERS.items():
        rows.append((f"Tier: {name}", f"${i} in / ${o} out per M tokens"))
    rows += [
        ("Local device draw", f"{config.LOCAL_DEVICE_WATTS} W while generating"),
        ("Electricity", f"₹{config.ELECTRICITY_INR_PER_KWH}/kWh"),
        ("Rented GPU", f"₹{config.GPU_INR_PER_HOUR}/hour"),
        ("GPU throughput", f"{config.GPU_OUTPUT_TOK_PER_SEC:g} output tok/s, "
                           f"{config.GPU_PREFILL_TOK_PER_SEC:g} prefill tok/s (single stream)"),
        ("GPU floor", f"{config.GPU_HOURS_PER_MONTH:g} hours/month (one GPU always on)"),
        ("Scale basis", "10,000 users × 1 proposal/month"),
        ("Pricing checked", "2026-10-08, platform.claude.com pricing page"),
    ]
    return rows
