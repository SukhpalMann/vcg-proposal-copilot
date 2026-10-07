"""List the claims a REAL model wrote that the verifier blocked, for the demo.

The deterministic mock plants a "35 percent" overclaim so the tests have
something to catch. That is fine for tests and wrong for a pitch: if someone asks
"did the AI really write that?", the answer must be yes. This script produces the
honest version: every statement the model drafted that the rule-based verifier
refused, with the exact text, what it cited, and why it was blocked.

    # from a run recorded with scripts/record_demo_run.py
    python scripts/capture_real_errors.py --run-id demo-local
    # or draft a fresh run now
    LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=... python scripts/capture_real_errors.py \
        --fixture abc_bank_lending_transformation.md

Writes docs/real_model_errors.md. Refuses mock runs, because their errors are planted.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from models.schemas import VerificationStatus  # noqa: E402
from pipeline.graph import continue_approved_pipeline, load_run, run_pipeline  # noqa: E402
from pipeline.qualification import record_decision  # noqa: E402
from services.vectorstore import VectorStore  # noqa: E402


def blocked_claims(state) -> list[dict]:
    results = state.get("_verification_results") or {}
    out = []
    for c in state.get("atomic_claims") or []:
        r = results.get(c.claim_id) or {}
        if r.get("status") != VerificationStatus.GAP:
            continue
        unresolved = list(r.get("unresolved_citations") or [])
        if unresolved:
            kind = "invented citation"
        elif not c.cited_evidence_ids:
            kind = "factual claim with no citation"
        elif c.numeric_tokens:
            kind = "figure not in the cited passage"
        else:
            kind = "not supported by the cited passage"
        out.append({
            "section": c.section_name,
            "claim": c.claim_text,
            "cited": c.cited_evidence_ids,
            "unresolved": unresolved,
            "kind": kind,
            "reason": r.get("reason", ""),
        })
    # the most persuasive demo examples first: a figure, then an invented id
    order = {"figure not in the cited passage": 0, "invented citation": 1,
             "factual claim with no citation": 2, "not supported by the cited passage": 3}
    return sorted(out, key=lambda e: order.get(e["kind"], 9))


def to_markdown(state, rows: list[dict]) -> str:
    provider = state.get("generation_provider")
    usage = state.get("model_usage") or []
    model = usage[0].get("model") if usage else config.LLM_MODEL
    lines = [
        "# Real model errors the verifier blocked",
        "",
        f"Run `{state.get('run_id')}` on `{state.get('rfp_filename')}`, provider "
        f"`{provider}`, model `{model}`, captured {time.strftime('%Y-%m-%d %H:%M')}.",
        "",
        f"{len(rows)} drafted statement(s) were blocked. Each was written by the model "
        "and refused by rule, not by another model.",
        "",
    ]
    for i, r in enumerate(rows, 1):
        lines += [
            f"## {i}. {r['kind'].capitalize()} ({r['section']})",
            "",
            f"> {r['claim']}",
            "",
            f"- Cited: {', '.join(r['cited']) or '(nothing)'}",
        ]
        if r["unresolved"]:
            lines.append(f"- Ids that resolve to no selected evidence: {', '.join(r['unresolved'])}")
        lines += [f"- Verifier: {r['reason']}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--run-id", help="reopen a saved run (e.g. demo-local)")
    g.add_argument("--fixture", default="abc_bank_lending_transformation.md")
    ap.add_argument("--out", default=str(ROOT / "docs" / "real_model_errors.md"))
    args = ap.parse_args(argv)

    config.ensure_dirs()
    VectorStore.ensure_seeded()
    if args.run_id:
        state = load_run(args.run_id)
    else:
        if config.LLM_PROVIDER == "mock":
            print("Set LLM_PROVIDER=ollama or anthropic: the mock's errors are planted.")
            return 2
        state = run_pipeline(str(config.FIXTURE_DIR / args.fixture), persist=True)
        record_decision(state, "BID", "Error capture", "Capturing real model output.")
        state = continue_approved_pipeline(state, persist=True)

    if state.get("generation_provider") in (None, "mock"):
        print(f"Run {state.get('run_id')} was drafted by the mock, whose errors are "
              "planted. Record a real run first (scripts/record_demo_run.py).")
        return 2

    rows = blocked_claims(state)
    md = to_markdown(state, rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md)
    print(md)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
