"""Phase 6 -- SQLite audit log + review decisions + run snapshot."""
from __future__ import annotations

from models.schemas import ReviewDecision
from pipeline import review
from services.persistence import Store


def test_audit_log_and_decisions_round_trip(tmp_path):
    db = Store(tmp_path / "a.sqlite")
    db.log("run-1", "intake", "ok", notes="hello")
    db.record_decision("run-1", "SEC-00", "partner", "APPROVED", "looks good")
    assert [e["stage"] for e in db.audit_trail("run-1")] == ["intake"]
    assert db.decisions("run-1")[0]["decision"] == "APPROVED"


def test_snapshot_round_trip(tmp_path):
    db = Store(tmp_path / "b.sqlite")
    db.save_snapshot("run-2", {"review_status": "PENDING", "warnings": ["w"]})
    got = db.load_snapshot("run-2")
    assert got["review_status"] == "PENDING"
    assert "run-2" in {r["run_id"] for r in db.list_runs()}


def test_pipeline_run_writes_audit_trail(abc_state):
    # abc_state fixture runs with persist=False; run one persisted pipeline here
    from pipeline.graph import continue_approved_pipeline, run_pipeline
    from pipeline.qualification import record_decision
    import config
    st = run_pipeline(str(config.FIXTURE_DIR / "abc_bank_lending_transformation.md"),
                      run_id="persisted-1", persist=True)
    record_decision(st, "BID", "fixture practice lead", "Test fixture")
    st = continue_approved_pipeline(st)
    from services.persistence import get_store
    trail = get_store().audit_trail("persisted-1")
    stages = {e["stage"] for e in trail}
    assert "intake" in stages and "build_traceability" in stages
    assert any(e["status"] == "PENDING" for e in trail)  # await_human_review


def test_run_can_be_resumed_with_full_working_state():
    from pipeline.graph import continue_approved_pipeline, load_run, run_pipeline
    from pipeline import review, export, qualification
    import config
    from models.schemas import ReviewDecision

    st = run_pipeline(str(config.FIXTURE_DIR / "abc_bank_lending_transformation.md"),
                      run_id="resume-1", persist=True)
    qualification.record_decision(st, "BID", "practice lead", "Proceed to proposal")
    st = continue_approved_pipeline(st)
    review.submit_section_decision(st, st["proposal_draft"].sections[0].section_id,
                                   "p", ReviewDecision.APPROVED, "ok")

    # reopen from persistence -- review progress + full state survive
    again = load_run("resume-1")
    assert again["proposal_draft"].sections[0].review_status == ReviewDecision.APPROVED
    assert again["selected_evidence"]           # full working state, not just a summary
    assert again["overall_traceability"]
    # can still drive review + export on the resumed state
    for s in again["proposal_draft"].sections:
        review.submit_section_decision(again, s.section_id, "p", ReviewDecision.APPROVED)
    for g in review.unresolved_gaps(again):
        review.override_gap(again, g.trace_id, "not claiming unsupported metric")
    review.approve_price(again, "partner", "quote-001", "Commercial response checked")
    ok, _ = review.can_export(again)
    assert ok and "## Executive Summary" in export.render_markdown(again)


def test_decision_is_recorded_to_db(abc_state):
    from services.persistence import get_store
    sec = abc_state["proposal_draft"].sections[0]
    review.submit_section_decision(abc_state, sec.section_id, "em",
                                   ReviewDecision.CHANGES_REQUESTED, "tighten")
    rows = get_store().decisions(abc_state["run_id"])
    assert rows and rows[-1]["decision"] == "CHANGES_REQUESTED"


def test_unpicklable_state_does_not_fail_the_run(monkeypatch):
    """A hot code reload leaves stale classes in session state; saving must not crash."""
    import pickle

    from pipeline import graph

    def boom(_obj):
        raise pickle.PicklingError("Can't pickle <class 'X'>: it's not the same object as X")

    monkeypatch.setattr(pickle, "dumps", boom)
    state = {"run_id": "stale-classes", "warnings": [], "execution_log": []}
    graph.save_run_state(state)                      # must not raise
    assert any("not saved for reopening" in w for w in state["warnings"])
