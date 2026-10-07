"""Measure drafting prompt v1 against v2 on the same tenders, with a real model.

The rubric rewards "a prompt pipeline you tested and improved". This is that
test. Each variant drafts the same tenders end to end; the deterministic
verifier then scores what the model wrote. Nothing here is judged by a model.

    # local (slow, private)
    LLM_PROVIDER=ollama LLM_MODEL=gemma3:latest python scripts/prompt_ab.py
    # hosted (fast)
    LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=... python scripts/prompt_ab.py
    # options
    python scripts/prompt_ab.py --fixtures abc_bank_lending_transformation.md --repeats 2

Writes docs/prompt_ab_results.md (a table for the deck) and .json (raw numbers).
It refuses to run on the mock: the mock's errors are planted, so measuring a
prompt against it would be meaningless.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from models.schemas import VerificationStatus  # noqa: E402
from pipeline.graph import continue_approved_pipeline, run_pipeline  # noqa: E402
from pipeline.qualification import record_decision  # noqa: E402
from services import costing  # noqa: E402
from services.vectorstore import VectorStore  # noqa: E402

DEFAULT_FIXTURES = [
    "abc_bank_lending_transformation.md",    # threshold the corpus cannot meet
    "merantau_sme_underwriting.md",          # partly covered, different segment
    "nutriva_supply_chain_planning.md",      # off-domain: no matching CV
]
VARIANTS = ("v1", "v2")


def measure(state) -> dict:
    """What the verifier found in one drafted run. Works on any completed state."""
    claims = state.get("atomic_claims") or []
    results = state.get("_verification_results") or {}
    factual = [c for c in claims if c.requires_verification]
    status = {c.claim_id: results.get(c.claim_id, {}).get("status") for c in claims}
    invented = sorted({cid for r in results.values()
                       for cid in (r.get("unresolved_citations") or [])})
    invented_count = sum(len(r.get("unresolved_citations") or []) for r in results.values())
    supported = sum(1 for c in factual if status[c.claim_id] == VerificationStatus.SUPPORTED)
    blocked = sum(1 for c in factual if status[c.claim_id] == VerificationStatus.GAP)
    orphan = sum(1 for c in factual if not c.cited_evidence_ids)
    unsupported_figures = sum(
        1 for c in factual
        if status[c.claim_id] == VerificationStatus.GAP and c.numeric_tokens)
    drafts = state.get("draft_sections") or {}
    usage = costing.session_cost(state.get("model_usage") or [])
    return {
        "factual_claims": len(factual),
        "substantiated": supported,
        "blocked": blocked,
        "substantiation_rate": round(supported / len(factual), 3) if factual else None,
        "invented_citations": invented_count,
        "invented_citation_examples": invented[:5],
        "uncited_factual_claims": orphan,
        "blocked_claims_with_figures": unsupported_figures,
        "gap_markers": sum(d.count("[EVIDENCE GAP") for d in drafts.values()),
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "model_seconds": usage["seconds"],
        "cost_inr": usage["api_equivalent_inr"],
    }


def run_once(fixture: str, variant: str) -> dict:
    config.DRAFT_PROMPT_VERSION = variant
    state = run_pipeline(str(config.FIXTURE_DIR / fixture), persist=False)
    record_decision(state, "BID", "Prompt A/B harness", "Measurement run, not a bid decision.")
    state = continue_approved_pipeline(state, persist=False)
    return measure(state)


def _sum(rows: list[dict], key: str) -> float:
    return sum((r.get(key) or 0) for r in rows)


def summarise(rows: list[dict]) -> dict:
    factual = _sum(rows, "factual_claims")
    return {
        "runs": len(rows),
        "factual_claims": int(factual),
        "substantiated": int(_sum(rows, "substantiated")),
        "blocked": int(_sum(rows, "blocked")),
        "substantiation_rate": round(_sum(rows, "substantiated") / factual, 3) if factual else None,
        "invented_citations": int(_sum(rows, "invented_citations")),
        "uncited_factual_claims": int(_sum(rows, "uncited_factual_claims")),
        "blocked_claims_with_figures": int(_sum(rows, "blocked_claims_with_figures")),
        "gap_markers": int(_sum(rows, "gap_markers")),
        "input_tokens": int(_sum(rows, "input_tokens")),
        "output_tokens": int(_sum(rows, "output_tokens")),
        "model_seconds": round(_sum(rows, "model_seconds"), 1),
        "cost_inr": round(_sum(rows, "cost_inr"), 3),
    }


def to_markdown(summary: dict, per_run: list[dict], meta: dict) -> str:
    keys = [
        ("Factual claims drafted", "factual_claims"),
        ("Substantiated", "substantiated"),
        ("Blocked by the verifier", "blocked"),
        ("Substantiation rate", "substantiation_rate"),
        ("Invented citation ids", "invented_citations"),
        ("Factual claims with no citation", "uncited_factual_claims"),
        ("Blocked claims carrying a figure", "blocked_claims_with_figures"),
        ("Evidence-gap markers written", "gap_markers"),
        ("Input tokens", "input_tokens"),
        ("Output tokens", "output_tokens"),
        ("Model time (s)", "model_seconds"),
        ("Token cost (Rs, at configured hosted rate)", "cost_inr"),
    ]
    lines = [
        "# Drafting prompt A/B: v1 vs v2",
        "",
        f"Provider `{meta['provider']}`, model `{meta['model']}`, "
        f"run {meta['when']}. Fixtures: {', '.join(meta['fixtures'])}; "
        f"repeats per fixture: {meta['repeats']}.",
        "",
        "Every number below comes from the rule-based verifier, not from a model.",
        "",
        "| Metric | v1 | v2 |",
        "|---|---|---|",
    ]
    for label, k in keys:
        a, b = summary["v1"].get(k), summary["v2"].get(k)
        if k == "substantiation_rate":
            a = f"{a:.0%}" if a is not None else "n/a"
            b = f"{b:.0%}" if b is not None else "n/a"
        lines.append(f"| {label} | {a} | {b} |")
    lines += ["", "## Per run", "",
              "| Variant | Fixture | Factual | Substantiated | Blocked | Invented ids | Uncited |",
              "|---|---|---|---|---|---|---|"]
    for r in per_run:
        lines.append(f"| {r['variant']} | {r['fixture']} | {r['factual_claims']} | "
                     f"{r['substantiated']} | {r['blocked']} | {r['invented_citations']} | "
                     f"{r['uncited_factual_claims']} |")
    lines += ["", "Lower is better for invented ids, uncited claims and blocked figures. "
              "A lower substantiation rate with fewer factual claims can still be an "
              "improvement: the prompt may have converted bluffs into gap markers.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fixtures", nargs="*", default=DEFAULT_FIXTURES)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "docs" / "prompt_ab_results"))
    args = ap.parse_args(argv)

    if config.LLM_PROVIDER == "mock":
        print("Refusing to run on the mock provider: its errors are planted, so a prompt "
              "comparison would measure nothing. Set LLM_PROVIDER=ollama or anthropic.")
        return 2

    config.ensure_dirs()
    VectorStore.ensure_seeded()
    per_run: list[dict] = []
    for variant in VARIANTS:
        for fixture in args.fixtures:
            for i in range(args.repeats):
                t0 = time.time()
                row = run_once(fixture, variant)
                row.update(variant=variant, fixture=fixture, repeat=i + 1)
                per_run.append(row)
                print(f"[{variant}] {fixture} #{i + 1}: factual={row['factual_claims']} "
                      f"ok={row['substantiated']} blocked={row['blocked']} "
                      f"invented_ids={row['invented_citations']} "
                      f"({time.time() - t0:.0f}s)")

    summary = {v: summarise([r for r in per_run if r["variant"] == v]) for v in VARIANTS}
    meta = {"provider": config.LLM_PROVIDER, "model": config.LLM_MODEL,
            "when": time.strftime("%Y-%m-%d %H:%M"), "fixtures": args.fixtures,
            "repeats": args.repeats}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(
        {"meta": meta, "summary": summary, "runs": per_run}, indent=2))
    out.with_suffix(".md").write_text(to_markdown(summary, per_run, meta))
    print("\n" + to_markdown(summary, per_run, meta))
    print(f"Saved {out.with_suffix('.md')} and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
