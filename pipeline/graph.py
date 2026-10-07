"""Orchestrator.

SPEC Section 0 fallback taken deliberately: a plain ordered list of Python
functions, each mutating one state dict and appending to ``execution_log``. The
two real branch points in SPEC Section 5 are handled inline:

  * validate_requirements -> ungrounded requirements are dropped, not passed on;
  * retrieval_decision (gap) -> optional_web_enrichment (a no-op when disabled).

The graph stops at ``await_human_review`` and returns state. Nothing downstream
of here (finalize, export) runs without human section approval -- see
pipeline/review.py.
"""
from __future__ import annotations

import uuid

import config
from pipeline import (
    claims,
    conflicts,
    drafting,
    extraction,
    intake,
    planning,
    qualification,
    ranking,
    retrieval,
    traceability,
    verification,
    web_enrichment,
)
from services.persistence import get_store
from services.llm import reset_usage_sink, set_usage_sink
from services.vectorstore import VectorStore
from state.graph_state import ProposalAgentState, new_state

# ordered stages up to the human gate
_STAGES = [
    ("intake", intake.run),
    ("decompose_rfp + validate_requirements", extraction.run),
    ("plan_response", planning.run),
    ("retrieve_internal", None),          # needs the store, handled specially
    ("rank_evidence", ranking.run),
    ("qualify", qualification.run),
    ("detect_conflicts", conflicts.run),
    ("optional_web_enrichment", web_enrichment.run),
    ("draft_sections", drafting.run),
    ("extract_atomic_claims", claims.run),
    ("deterministic_verify", None),       # needs the semantic fn
    ("build_traceability", traceability.run),
]


def run_pipeline(
    rfp_path: str,
    *,
    run_id: str | None = None,
    web_search: bool = False,
    store: VectorStore | None = None,
    persist: bool = True,
) -> ProposalAgentState:
    run_id = run_id or uuid.uuid4().hex
    store = store or VectorStore.load()
    db = get_store() if persist else None

    state = new_state(run_id=run_id, rfp_path=rfp_path, web_search_enabled=web_search)
    state["generation_provider"] = config.LLM_PROVIDER
    state["_semantic_fn"] = store.semantic_similarity

    def _audit(stage: str, status: str, notes: str = "") -> None:
        if db:
            db.log(run_id, stage, status, notes=notes)

    usage_token = set_usage_sink(state["model_usage"])
    try:
      for name, fn in _STAGES:
        try:
            if name == "retrieve_internal":
                state = retrieval.run(state, store=store)
            elif name == "deterministic_verify":
                state = verification.run(state, semantic_fn=store.semantic_similarity)
            else:
                state = fn(state)
            last = state["execution_log"][-1] if state["execution_log"] else {}
            _audit(name, "ok", str(last.get("detail", "")))
            if db:
                db.save_snapshot(run_id, _snapshot(state))
            if name == "qualify":
                if db:
                    save_run_state(state, db)
                return state
        except Exception as exc:  # keep the run inspectable on failure
            state.setdefault("errors", []).append(f"{name}: {exc}")
            state["execution_log"].append({"stage": name, "status": "error", "detail": str(exc)})
            _audit(name, "error", str(exc))
            raise
    finally:
      reset_usage_sink(usage_token)

    raise RuntimeError("Qualification stage was not reached")


def continue_approved_pipeline(state: ProposalAgentState, *, persist: bool = True) -> ProposalAgentState:
    """Resume after a recorded BID decision; never draft a NO_BID opportunity."""
    decision = state.get("practice_lead_decision") or {}
    if decision.get("decision") != "BID":
        raise PermissionError("Practice lead must approve BID before drafting")
    store = VectorStore.load()
    state["_semantic_fn"] = store.semantic_similarity
    db = get_store() if persist else None
    after_gate = False
    usage_token = set_usage_sink(state["model_usage"])
    try:
      for name, fn in _STAGES:
        if name == "qualify":
            after_gate = True
            continue
        if not after_gate:
            continue
        try:
            if name == "deterministic_verify":
                state = verification.run(state, semantic_fn=store.semantic_similarity)
            else:
                state = fn(state)
            if db:
                last = state["execution_log"][-1] if state["execution_log"] else {}
                db.log(state["run_id"], name, "ok", notes=str(last.get("detail", "")))
                db.save_snapshot(state["run_id"], _snapshot(state))
        except Exception as exc:
            state.setdefault("errors", []).append(f"{name}: {exc}")
            state["execution_log"].append({"stage": name, "status": "error", "detail": str(exc)})
            if db:
                db.log(state["run_id"], name, "error", notes=str(exc))
            raise
    finally:
      reset_usage_sink(usage_token)
    state["review_status"] = "PENDING"
    if db:
        db.log(state["run_id"], "await_human_review", "PENDING", notes="awaiting section-level approval")
        save_run_state(state, db)
    return state


def save_run_state(state: ProposalAgentState, db=None) -> None:
    """Persist the full working state so a run can be reopened and continued
    (the manual checkpointer stand-in -- SPEC Section 0)."""
    import pickle

    db = db or get_store()
    snapshot = {k: v for k, v in state.items() if k != "_semantic_fn"}
    try:
        blob = pickle.dumps(snapshot)
    except Exception as exc:
        # Saving is what lets a run be REOPENED later; it must never fail the run
        # itself. The usual cause is a hot code reload on Streamlit Cloud: the
        # session still holds objects built from the previous version of a class,
        # and pickle refuses to mix the two. The run carries on; only "Reopen
        # run" is unavailable for it until the app is rebooted.
        note = (f"Run not saved for reopening ({type(exc).__name__}). The app's code "
                "was updated during this session; reboot the app to restore saving.")
        if note not in state.setdefault("warnings", []):
            state["warnings"].append(note)
        blob = None
    if blob is not None:
        db.save_state_blob(state["run_id"], blob)
    db.save_snapshot(state["run_id"], _snapshot(state))


def load_run(run_id: str, store: VectorStore | None = None) -> ProposalAgentState:
    """Reopen a persisted run with a fully working state (review / regenerate /
    export all continue to function)."""
    import pickle

    db = get_store()
    blob = db.load_state_blob(run_id)
    if blob is None:
        raise KeyError(f"no persisted state for run {run_id}")
    state: ProposalAgentState = pickle.loads(blob)
    store = store or VectorStore.load()
    state["_semantic_fn"] = store.semantic_similarity
    return state


def _snapshot(state: ProposalAgentState) -> dict:
    keep = {
        "run_id", "rfp_filename", "warnings", "errors", "retrieval_gaps",
        "requirement_validation_errors", "review_status", "section_review_status",
        "execution_log", "procedural_checklist", "human_input_requirements",
        "qualification", "practice_lead_decision", "price_approval", "model_usage",
        "generation_provider",
    }
    out = {k: state.get(k) for k in keep}
    if state.get("proposal_draft") is not None:
        out["proposal_draft"] = state["proposal_draft"].model_dump()
    if state.get("overall_traceability"):
        out["overall_traceability"] = [e.model_dump() for e in state["overall_traceability"]]
    return out
