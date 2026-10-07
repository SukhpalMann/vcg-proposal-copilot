"""A true statement below the tender's bar must not satisfy the requirement.

ABC Bank asks for "greater than 30 percent" turnaround improvement. The corpus
records 18 percent. The 18 percent statement is substantiated as a claim, but the
requirement it is mapped to must not roll up as SUPPORTED, and the row must say why.
Also checks that requirement rows with no drafted claim are no longer reported
as PARTIAL (they were never verified).
"""
from pipeline.traceability import _threshold_shortfall


def test_shortfall_detected_below_bar():
    req = "Evidence of greater than 30 percent turnaround-time improvement"
    msg = _threshold_shortfall(req, "VCG reduced turnaround time by 18 percent")
    assert msg and "18 percent" in msg and "30" in msg


def test_no_shortfall_when_bar_met():
    req = "Evidence of greater than 30 percent turnaround-time improvement"
    assert _threshold_shortfall(req, "VCG reduced turnaround time by 35 percent") is None


def test_no_threshold_means_no_check():
    req = "Demonstrated experience redesigning retail lending operations"
    assert _threshold_shortfall(req, "VCG reduced turnaround time by 18 percent") is None


def test_abc_rollup_and_unverified_rows(abc_state):
    summary = abc_state["requirement_summary"]
    bar = [s for s in summary if "greater than 30 percent" in s["requirement"]]
    assert bar, "fixture should contain the >30 percent requirement"
    assert all(s["rollup_status"] != "SUPPORTED" for s in bar)

    unclaimed = [e for e in abc_state["overall_traceability"] if e.claim_id == "(none)"]
    assert unclaimed
    assert all(e.verification_status.value != "PARTIAL" for e in unclaimed)
