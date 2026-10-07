"""Deterministic CLI demo -- no live web, no API key required (SPEC Section 0).

    python scripts/seed_corpus.py
    python scripts/run_demo.py [fixture_name]

Prints the hero moment: the 18% claim passes as SUPPORTED, the 35% claim is
caught as GAP, in the Requirement -> Evidence -> Draft traceability matrix.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from pipeline.graph import continue_approved_pipeline, run_pipeline  # noqa: E402
from pipeline import export, qualification  # noqa: E402
from pipeline.traceability import evidence_display_id  # noqa: E402
from services.vectorstore import VectorStore  # noqa: E402


def _short(text: str, n: int = 58) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def main() -> int:
    config.ensure_dirs()
    VectorStore.ensure_seeded()          # self-seed so no manual step is required
    fixture = sys.argv[1] if len(sys.argv) > 1 else "abc_bank_lending_transformation.md"
    path = config.FIXTURE_DIR / fixture
    if not path.exists():
        print(f"no such fixture: {path}\navailable: "
              f"{[p.name for p in config.FIXTURE_DIR.glob('*.md')]}")
        return 1

    print(f"\n=== RUN: {fixture} ===\n")
    state = run_pipeline(str(path), web_search=False)
    qualification.record_decision(state, "BID", "demo practice lead",
                                  "Prepared demonstration fixture")
    state = continue_approved_pipeline(state)

    print("--- Execution log ---")
    for step in state["execution_log"]:
        print(f"  [{step['status']:>7}] {step['stage']}: {step.get('detail', '')}")

    if state["warnings"]:
        print("\n--- Warnings ---")
        for w in state["warnings"]:
            print(f"  ! {w}")

    if state["requirement_validation_errors"]:
        print("\n--- Ungrounded requirements (dropped) ---")
        for e in state["requirement_validation_errors"]:
            print(f"  x {e}")

    print("\n--- Procedural checklist (never becomes prose) ---")
    for p in state["procedural_checklist"]:
        print(f"  [ ] {p}")

    print("\n--- Evidence: selected vs rejected (sample) ---")
    for cid, evs in list(state["selected_evidence"].items())[:4]:
        item = next(c for c in state["checklist"] if c.checklist_id == cid)
        print(f"  {cid}  «{_short(item.requirement_text, 50)}»")
        for ev in evs[:2]:
            print(f"      SELECTED  {ev.source_id:<15} score={ev.relevance_score:.2f}  {ev.reasoning}")
        for ev in state["rejected_evidence"].get(cid, [])[:2]:
            print(f"      rejected  {ev.source_id:<15} {ev.rejection_reason}")

    if state["evidence_conflicts"]:
        print("\n--- Conflicts surfaced (never auto-resolved) ---")
        for c in state["evidence_conflicts"]:
            print(f"  {c.conflict_id} [{c.conflict_type}] {c.description}")

    print("\n" + "=" * 100)
    print("REQUIREMENT  ->  EVIDENCE  ->  DRAFT CLAIM  ->  STATUS   (traceability matrix)")
    print("=" * 100)
    hdr = f"{'Requirement':<40} {'Evidence':<16} {'Draft claim':<46} {'Status':<14} Conf"
    print(hdr)
    print("-" * len(hdr))
    for e in state["overall_traceability"]:
        if not e.claim_text or e.claim_text == "(no drafted claim)":
            ev = evidence_display_id(e.matched_evidence_id)
            print(f"{_short(e.rfp_requirement, 39):<40} {ev:<16} {'(no drafted claim)':<46} "
                  f"{e.verification_status.value:<14} {e.confidence_score:.2f}")
            continue
        ev = evidence_display_id(e.matched_evidence_id)
        print(f"{_short(e.rfp_requirement, 39):<40} {ev:<16} {_short(e.claim_text, 45):<46} "
              f"{e.verification_status.value:<14} {e.confidence_score:.2f}")

    draft = state["proposal_draft"]
    print("-" * len(hdr))
    print(f"TOTALS: SUPPORTED={draft.supported_claim_count}  "
          f"PARTIAL={draft.partial_claim_count}  GAP={draft.gap_claim_count}  "
          f"sections={len(draft.sections)}   (distinct drafted claims)")
    from collections import Counter
    unclaimed = Counter(e.verification_status.value for e in state["overall_traceability"]
                        if e.claim_id == "(none)")
    if unclaimed:
        print("REQUIREMENTS WITH NO DRAFTED CLAIM (never verified): "
              + "  ".join(f"{k}={v}" for k, v in sorted(unclaimed.items())))

    # hero assertion -- only meaningful for the happy-path fixture
    rows = state["overall_traceability"]
    is_hero = fixture.startswith("abc_bank")
    got_18 = any("18 percent" in r.claim_text and r.verification_status.value == "SUPPORTED"
                 for r in rows)
    got_35 = any("35 percent" in r.claim_text and r.verification_status.value == "GAP"
                 for r in rows)
    if is_hero:
        print("\nHERO CHECK:")
        print(f"  18% claim SUPPORTED : {'PASS' if got_18 else 'FAIL'}")
        print(f"  35% claim caught GAP: {'PASS' if got_35 else 'FAIL'}")

    ok, reasons = export.can_export(state)
    print(f"\nExport gate (pre-review): {'OPEN' if ok else 'BLOCKED'}")
    for r in reasons:
        print(f"  - {r}")
    print("  (sections require human approval in the Streamlit Review tab before export)\n")
    if is_hero:
        return 0 if (got_18 and got_35) else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
