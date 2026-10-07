"""Proposal Copilot -- application interface (SPEC Phase 7).

Tabs:
    Overview -> Traceability -> Evidence -> Draft -> Requirements -> Execution -> Review

Overview reports the state of the run and whether it may be released.
Traceability is the evidentiary record: for any claim, the requirement it
addresses, the quotation located in the RFP, the evidence passage it was drawn
from, the deterministic verification result, and the reviewer decision.

Presentation tokens and components are defined in ui.py.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd
import streamlit as st

import config
import ui
from models.schemas import ReviewDecision, VerificationStatus
from pipeline import export, qualification, review
from pipeline.graph import continue_approved_pipeline, load_run, run_pipeline
from pipeline.traceability import evidence_display_id
from services.persistence import get_store
from services.vectorstore import VectorStore

st.set_page_config(page_title="Proposal Copilot", layout="wide",
                   initial_sidebar_state="expanded")
config.ensure_dirs()


@st.cache_resource(show_spinner="Preparing the evidence base…")
def _bootstrap():
    """First-load setup so a hosted deployment requires no manual seed step."""
    VectorStore.ensure_seeded()
    return True


_bootstrap()
st.markdown(ui.CSS, unsafe_allow_html=True)
H = {"unsafe_allow_html": True}


# --------------------------------------------------------------------------- #
# Control panel
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("#### Proposal Copilot")
    st.markdown('<div class="note">RFP to source-grounded, review-ready proposal</div>', **H)
    if config.LLM_PROVIDER == "ollama":
        st.caption(f"Local AI: {config.LLM_MODEL}. Documents stay on this machine.")
    elif config.LLM_PROVIDER == "mock":
        st.warning("Simulation mode: the deterministic mock is for testing, not a live AI demo.")
    else:
        st.warning("Hosted model mode: documents are sent to the configured provider.")
    st.divider()

    fixtures = sorted(p.name for p in config.FIXTURE_DIR.glob("*.md"))
    choice = st.selectbox(
        "Sample RFP", fixtures,
        help="Four scenarios: a standard engagement, a capability gap, a "
             "procurement-governed submission, and an RFP with no stated "
             "evaluation criteria.")
    uploaded = st.file_uploader("Or upload a document", type=["md", "txt", "pdf"])
    web = st.checkbox("Include external market context", value=False,
                      disabled=config.LLM_PROVIDER in {"ollama", "mock"},
                      help="Supplies industry background only. External sources are "
                           "never cited as evidence of the firm's experience.")

    if st.button("Assess bid fit", type="primary", use_container_width=True):
        if uploaded is not None:
            dest = config.DATA_DIR / f"upload_{uploaded.name}"
            dest.write_bytes(uploaded.getbuffer())
            rfp_path = str(dest)
        else:
            rfp_path = str(config.FIXTURE_DIR / choice)
        with st.spinner("Processing…"):
            try:
                st.session_state.state = run_pipeline(
                    rfp_path, web_search=web)
                st.session_state.rfp_name = choice if uploaded is None else uploaded.name
            except Exception as exc:                       # keep the interface usable
                st.session_state.state = None
                st.error(f"Processing failed: {exc}")

    if st.session_state.get("state"):
        s = st.session_state.state
        if s.get("proposal_draft"):
            d = s["proposal_draft"]
            st.markdown(
                f'<div class="pills">{ui.pill("SUPPORTED", str(d.supported_claim_count))}'
                f'{ui.pill("PARTIAL", str(d.partial_claim_count))}'
                f'{ui.pill("GAP", str(d.gap_claim_count))}</div>', **H)
            st.markdown(f'<div class="note">Run {s["run_id"][:12]} · '
                        f'{len(s["overall_traceability"])} traceability records</div>', **H)
        else:
            st.caption(f"Run {s['run_id'][:12]} · awaiting practice-lead decision")
        if s.get("generation_provider") == "mock" and config.LLM_PROVIDER == "ollama":
            st.caption("This saved run was prepared in simulation mode; section regeneration uses local AI.")

    st.divider()
    st.markdown('<div class="sec-h">Saved runs</div>', **H)
    try:
        prev = get_store().list_runs()
    except Exception:
        prev = []
    if prev:
        pick = st.selectbox("Saved runs", [r["run_id"] for r in prev],
                            format_func=lambda r: r[:16], label_visibility="collapsed")
        if st.button("Reopen run", use_container_width=True):
            try:
                st.session_state.state = load_run(pick)
                st.session_state.rfp_name = st.session_state.state.get("rfp_filename", pick)
                st.rerun()
            except Exception as exc:
                st.error(f"Unable to reopen: {exc}")
    else:
        st.markdown('<div class="note">No saved runs.</div>', **H)


# --------------------------------------------------------------------------- #
# Entry state
# --------------------------------------------------------------------------- #
state = st.session_state.get("state")
if state and not state.get("proposal_draft"):
    q = state.get("qualification") or {}
    st.title("Bid qualification")
    st.caption("Screening aid based on selected firm evidence. A practice lead makes the decision.")
    st.metric("Evidence fit", "Unscored" if q.get("score") is None else f"{q['score']}%")
    st.write(q.get("reason", ""))
    if q.get("crm_client_match") is False:
        st.warning("The sample CRM client differs from the RFP client. Confirm the opportunity record.")
    with st.expander("Prototype system inputs"):
        st.caption("Local fictional sample files represent CRM, HR, rate-card, and time/billing inputs. No live integrations are claimed.")
        for name, record in (state.get("system_inputs") or {}).items():
            st.write(f"**{name.replace('_', ' ').title()}** — {record.get('source', 'sample')}")
    st.write(f"{q.get('covered_requirements', 0)} of {q.get('evidence_requirements', 0)} "
             "evidence requirements have selected passages.")
    if q.get("mandatory_gap_ids"):
        st.error("Mandatory evidence gaps: " + ", ".join(q["mandatory_gap_ids"]))
    decision = state.get("practice_lead_decision") or {}
    if decision.get("decision") == "NO_BID":
        st.warning("No-bid decision recorded. No proposal draft was generated.")
        st.stop()
    lead = st.text_input("Practice lead name", key="qual_lead")
    rationale = st.text_area("Decision rationale", key="qual_reason")
    b1, b2 = st.columns(2)
    if b1.button("Approve bid and generate draft", type="primary"):
        try:
            qualification.record_decision(state, "BID", lead, rationale)
            with st.spinner("Drafting after practice-lead approval…"):
                st.session_state.state = continue_approved_pipeline(state)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if b2.button("Record no-bid"):
        try:
            qualification.record_decision(state, "NO_BID", lead, rationale)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    st.stop()
if not state:
    st.markdown(
        """
        <div class="eyebrow">VCG · Proposal Copilot</div>
        <div class="hero-h1">Source-grounded proposals, verifiable claim by claim.</div>
        <div class="hero-sub">
          The system converts an inbound request for proposal into a review-ready
          document in which every factual statement is linked to the internal
          evidence it was drawn from. Statements the evidence base does not
          substantiate are identified and flagged before the document reaches a
          reviewer.
        </div>
        """, **H)
    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        '<div class="card"><h4>Requirement-to-evidence traceability</h4>'
        '<p>Each drafted statement is linked to the requirement it addresses and '
        'to the source passage it was written from. Evidence candidates that were '
        'considered and rejected are retained, together with the reason for their '
        'exclusion.</p></div>', **H)
    c2.markdown(
        '<div class="card"><h4>Deterministic verification</h4>'
        '<p>Numeric, attribution and contextual consistency are assessed by rule, '
        'not by a language model evaluating its own output. A figure contradicted '
        'by the cited evidence cannot be recorded as substantiated.</p></div>', **H)
    c3.markdown(
        '<div class="card"><h4>Mandatory review gate</h4>'
        '<p>No execution path produces an export without reviewer approval of every '
        'section. Unsubstantiated statements block release until they are resolved '
        'or formally overridden with a recorded justification.</p></div>', **H)

    st.write("")
    st.markdown(
        '<div class="flow">'
        '<span>Ingest</span><span>Extract</span><span>Validate</span>'
        '<span>Retrieve</span><span>Qualify</span><span>Draft</span>'
        '<span>Decompose</span><span>Verify</span>'
        '<span class="gate">Review</span><span>Release</span></div>', **H)
    st.markdown(
        '<div class="cta">Select a sample RFP in the panel on the left and choose '
        '<b>Assess bid fit</b>. The <b>ABC Bank</b> scenario contains a '
        'performance claim that the evidence base does not substantiate.</div>', **H)
    st.stop()


# --------------------------------------------------------------------------- #
# Derived figures
# --------------------------------------------------------------------------- #
draft = state["proposal_draft"]
entries = state["overall_traceability"]
summary = state.get("requirement_summary", [])
export_ok, export_reasons = review.can_export(state)

n_selected = sum(len(v) for k, v in state["selected_evidence"].items() if k != "__pool__")
n_rejected = sum(len(v) for v in state["rejected_evidence"].values())
n_gap_rows = sum(1 for e in entries if e.verification_status == VerificationStatus.GAP)
n_approved = sum(1 for s in draft.sections if s.review_status == ReviewDecision.APPROVED)
coverage = Counter(s["rollup_status"] for s in summary)
usage = Counter(evidence_display_id(e.matched_evidence_id)
                for e in entries if e.matched_evidence_id)

_client = (state["rfp_data"].client or "Untitled RFP").rstrip(".")
st.markdown(
    f'<div class="eyebrow">Proposal · {state.get("review_status", "PENDING").replace("_", " ").title()}</div>'
    f'<div class="hero-h1" style="font-size:1.7rem;margin:.25rem 0 .5rem;">{_client}</div>'
    f'<div class="note">{st.session_state.get("rfp_name", "")} &nbsp;·&nbsp; '
    f'Run {state["run_id"][:12]} &nbsp;·&nbsp; {len(state["checklist"])} evidence '
    f'requirements &nbsp;·&nbsp; {len(state["procedural_checklist"])} procedural '
    f'requirements</div>', **H)
st.write("")

tab_over, tab_trace, tab_ev, tab_draft, tab_req, tab_exec, tab_review = st.tabs(
    ["Overview", "Traceability", "Evidence", "Draft",
     "Requirements", "Execution", "Review"]
)


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
with tab_over:
    st.markdown(ui.verdict(
        export_ok,
        "Cleared for release" if export_ok else "Not cleared for release",
        "All sections have been approved and no unsubstantiated claims remain."
        if export_ok else "; ".join(export_reasons) + "."), **H)
    st.write("")

    st.markdown(ui.tiles([
        ("Requirements", len(summary), "extracted and validated", ui.BRAND),
        ("Claims assessed", len(entries), "atomic, independently checked", ui.BRAND),
        ("Substantiated", draft.supported_claim_count, "confirmed against evidence",
         ui.STATUS["SUPPORTED"]["fill"]),
        ("Unsubstantiated", n_gap_rows, "blocking release", ui.STATUS["GAP"]["fill"]),
        ("Sections approved", f"{n_approved} of {len(draft.sections)}", "reviewer sign-off",
         ui.STATUS["SUPPORTED"]["fill"] if n_approved == len(draft.sections)
         else ui.STATUS["PARTIAL"]["fill"]),
        ("Evidence selected", n_selected, f"of {n_selected + n_rejected} retrieved",
         ui.MUTED),
    ]), **H)

    st.write("")
    left, right = st.columns([1.15, 1])

    with left:
        st.markdown('<div class="sec-h">Requirement disposition</div>', **H)
        chart = ui.coverage_bar(coverage)
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)
        st.markdown(
            '<div class="pills">' + "".join(
                ui.pill(k, f"{ui.STATUS[k]['label']} · {coverage[k]}")
                for k in ui.STATUS_ORDER if coverage.get(k)
            ) + "</div>", **H)
        st.markdown('<div class="note">Disposition of every requirement extracted '
                    'from the source document. Requirements marked not addressed '
                    'produced no drafted statement.</div>', **H)

    with right:
        st.markdown('<div class="sec-h">Evidence cited in the draft</div>', **H)
        chart = ui.evidence_bar(usage)
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.markdown('<div class="note">No evidence was cited by any '
                        'statement.</div>', **H)
        st.markdown(f'<div class="note">{n_selected} passages met the selection '
                    f'threshold from {n_selected + n_rejected} retrieved candidates. '
                    f'Each exclusion is recorded with a reason under Evidence.</div>', **H)

    st.write("")
    st.markdown('<div class="sec-h">Section readiness</div>', **H)
    rows = "".join(
        f'<tr><td><b>{s.title}</b>{" (edited)" if s.human_edited else ""}</td>'
        f'<td class="num">{len(s.claim_ids)}</td>'
        f'<td>{ui.pill("GAP" if s.has_gaps else "SUPPORTED", "Evidence gap" if s.has_gaps else "Complete")}</td>'
        f'<td>{ui.pill("SUPPORTED" if s.review_status == ReviewDecision.APPROVED else "NO_ROW", s.review_status.value.replace("_", " ").title())}</td></tr>'
        for s in draft.sections)
    st.markdown(
        '<table class="readiness"><tr><th>Section</th><th style="text-align:right">Claims</th>'
        '<th>Content</th><th>Review</th></tr>' + rows + "</table>", **H)

    if state["warnings"]:
        st.write("")
        st.markdown('<div class="sec-h">Escalated for review</div>', **H)
        for w in state["warnings"]:
            st.warning(w)


# --------------------------------------------------------------------------- #
# Traceability
# --------------------------------------------------------------------------- #
with tab_trace:
    # The contrast pair is derived, not hard-coded: the best-substantiated
    # numeric claim beside the numeric claim the evidence refused. On the demo
    # tender these are the 18% and 35% figures; on any other tender the panel
    # still works because nothing here matches on a literal value.
    def _numeric(e):
        return any(ch.isdigit() for ch in e.claim_text) and e.claim_id != "(none)"

    hero_18 = max((e for e in entries
                   if e.verification_status == VerificationStatus.SUPPORTED and _numeric(e)),
                  key=lambda e: e.confidence_score, default=None)
    hero_35 = next((e for e in entries
                    if e.verification_status == VerificationStatus.GAP and _numeric(e)), None)
    if hero_18 or hero_35:
        hc1, hc2 = st.columns(2)
        if hero_18:
            hc1.markdown(ui.verdict(
                True, "Substantiated",
                f'“{hero_18.claim_text}”<br><span style="color:{ui.MUTED}">Traced to '
                f'{evidence_display_id(hero_18.matched_evidence_id)}. The figure is '
                f'confirmed within the context of the cited passage.</span>'), **H)
        if hero_35:
            hc2.markdown(ui.verdict(
                False, "Unsubstantiated",
                f'“{hero_35.claim_text}”<br><span style="color:{ui.MUTED}">'
                f'{hero_35.verification_reason}</span>'), **H)
        st.markdown('<div class="note">Both statements were produced by the same '
                    'drafting step. Only one is supported by the evidence base, and '
                    'the distinction is established by rule rather than by '
                    'judgement.</div>', **H)
        st.write("")

    f1, f2 = st.columns([2, 1])
    show = f1.multiselect(
        "Filter by verification result", [s.value for s in VerificationStatus],
        default=["SUPPORTED", "PARTIAL", "GAP", "NARRATIVE", "HUMAN_INPUT"],
        format_func=lambda v: ui.STATUS.get(v, ui.STATUS["NO_ROW"])["label"])
    only_reqs = f2.checkbox("Requirement-linked records only", value=True)

    view = [e for e in entries
            if e.verification_status.value in show
            and (not only_reqs or e.requirement_id != "-")]

    df = pd.DataFrame([{
        "Result": ui.STATUS.get(e.verification_status.value,
                                ui.STATUS["NO_ROW"])["label"],
        "Requirement": e.rfp_requirement[:72],
        "Evidence": evidence_display_id(e.matched_evidence_id),
        "Statement": e.claim_text[:86],
        "Confidence": round(e.confidence_score, 2),
        "Reviewer": e.reviewer_decision.value.replace("_", " ").title(),
    } for e in view])

    if not df.empty:
        _tint = {ui.STATUS[k]["label"]: ui.STATUS[k]["tint"] for k in ui.STATUS}
        st.dataframe(
            df.style.apply(
                lambda r: [f"background-color: {_tint.get(r['Result'], '')}"] * len(r),
                axis=1),
            use_container_width=True, hide_index=True, height=340,
            column_config={"Confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0.0, max_value=1.0, format="%.2f")})
    else:
        st.markdown('<div class="note">No records match the current filter.</div>', **H)

    st.write("")
    st.markdown('<div class="sec-h">Claim inspection</div>', **H)
    if view:
        labels = [f"{ui.STATUS.get(e.verification_status.value, ui.STATUS['NO_ROW'])['label']}"
                  f"  —  {e.claim_text[:88]}" for e in view]
        idx = st.selectbox("Statement", range(len(labels)),
                           format_func=lambda i: labels[i], label_visibility="collapsed")
        e = view[idx]

        st.markdown(
            f'<div class="pills">{ui.pill(e.verification_status.value)}'
            f'{ui.pill("NO_ROW", "Confidence " + format(e.confidence_score, ".2f"))}'
            f'{ui.pill("NO_ROW", e.draft_section)}</div>', **H)

        st.markdown(ui.chain_step(
            f"1 — Requirement ({e.requirement_id})", e.rfp_requirement), **H)
        if e.requirement_source_span:
            st.markdown(ui.chain_step(
                "2 — Quotation located in the source document",
                f'“{e.requirement_source_span.quote}”', "quote"), **H)
        if e.matched_chunk_text:
            st.markdown(ui.chain_step(
                f"3 — Evidence passage cited ({e.matched_evidence_id})",
                e.matched_chunk_text.strip()[:700].replace("\n", "<br>"), "mono"), **H)
        else:
            st.markdown(ui.chain_step(
                "3 — Evidence passage cited",
                "None. This statement cites no selected evidence."), **H)
        st.markdown(ui.chain_step("4 — Drafted statement", e.claim_text), **H)

        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Semantic", f"{e.semantic_similarity:.2f}")
        v2.metric("Lexical", f"{e.lexical_overlap:.2f}")
        v3.metric("Numeric", {True: "Consistent", False: "Contradicted",
                              None: "Not applicable"}[e.numeric_match])
        v4.metric("Attribution", "Valid" if e.attribution_valid else "Mismatch")
        st.markdown(ui.chain_step("5 — Verification result", e.verification_reason), **H)

    st.write("")
    st.download_button("Download traceability matrix (CSV)",
                       export.traceability_csv(state),
                       file_name=f"traceability_{state['run_id'][:8]}.csv",
                       mime="text/csv")


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #
with tab_ev:
    if state["evidence_conflicts"]:
        st.error("Conflicting evidence identified. Conflicts are surfaced for "
                 "resolution and are never reconciled automatically.")
        for c in state["evidence_conflicts"]:
            st.markdown(f"- `{c.conflict_id}` **{c.conflict_type}** — {c.description}")

    st.markdown(f'<div class="note">{n_selected} passages met the selection threshold '
                f'from {n_selected + n_rejected} retrieved candidates. Every exclusion '
                f'carries a recorded reason.</div>', **H)
    st.write("")

    checklist_by_id = {c.checklist_id: c for c in state["checklist"]}
    for cid, sel in state["selected_evidence"].items():
        if cid == "__pool__":
            continue
        item = checklist_by_id.get(cid)
        if item is None:
            continue
        with st.expander(f"{item.requirement_text}   ·   {item.target_section}"):
            st.markdown(
                f'<div class="pills">'
                f'{ui.pill("SUPPORTED" if sel else "GAP", "Evidence selected" if sel else "No qualifying evidence")}'
                f'{ui.pill("NO_ROW", str(len(state["rejected_evidence"].get(cid, []))) + " excluded")}'
                f'</div>', **H)
            st.markdown(f'<div class="note">{item.evidence_need}</div>', **H)
            for e in sel:
                st.markdown(f"**{e.source_id}** — {e.reasoning}")
                st.code(e.chunk_text.strip()[:600])
            if not sel:
                st.markdown("**No candidate met the selection threshold. Recorded as "
                            "an evidence gap.**")
            rej = state["rejected_evidence"].get(cid, [])
            if rej:
                st.markdown("Excluded candidates:")
                for e in rej[:6]:
                    st.markdown(f"- `{e.source_id}` — {e.rejection_reason}")


# --------------------------------------------------------------------------- #
# Draft
# --------------------------------------------------------------------------- #
with tab_draft:
    for sec in draft.sections:
        badges = ui.pill("GAP" if sec.has_gaps else "SUPPORTED",
                         "Evidence gap" if sec.has_gaps else "Complete")
        if sec.human_edited:
            badges += ui.pill("FORWARD_LOOKING", "Manually edited")
        st.markdown(f"#### {sec.title}")
        st.markdown(f'<div class="pills">{badges}</div>', **H)
        st.markdown(sec.content_markdown)
        st.divider()


# --------------------------------------------------------------------------- #
# Requirements
# --------------------------------------------------------------------------- #
with tab_req:
    rd = state["rfp_data"]
    if state["requirement_validation_errors"]:
        st.error("The following requirements were discarded because the quotation "
                 "attributed to them could not be located in the source document.")
        for e in state["requirement_validation_errors"]:
            st.markdown(f"- {e}")

    st.markdown('<div class="sec-h">Extracted requirements</div>', **H)
    st.markdown('<div class="note">Each requirement is retained only where the '
                'quotation attributed to it is locatable in the source document.</div>', **H)
    st.dataframe(pd.DataFrame([{
        "ID": r.requirement_id, "Requirement": r.text,
        "Category": r.category.value.title(),
        "Handling": r.handling.value.replace("_", " ").title(),
        "Mandatory": r.mandatory, "Confidence": round(r.extraction_confidence, 2),
        "Source quotation": r.source_span.quote,
    } for r in rd.requirements]), use_container_width=True, hide_index=True,
        column_config={"Confidence": st.column_config.ProgressColumn(
            "Confidence", min_value=0.0, max_value=1.0, format="%.2f")})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec-h">Procedural requirements</div>', **H)
        st.markdown('<div class="note">Tracked for compliance. These are excluded '
                    'from proposal prose.</div>', **H)
        for p in state["procedural_checklist"] or ["None recorded."]:
            st.checkbox(p, key=f"proc_{p[:30]}", value=False)
    with c2:
        st.markdown('<div class="sec-h">Requires human input</div>', **H)
        for h in state["human_input_requirements"] or ["None recorded."]:
            st.markdown(f"- {h}")
        caps = [r.text for r in rd.requirements if r.handling.value == "CAPABILITY_GAP"]
        if caps:
            st.markdown('<div class="sec-h">Capability gaps</div>', **H)
            st.markdown('<div class="note">Escalated for a go / no-go decision.</div>', **H)
            for c in caps:
                st.markdown(f'{ui.pill("GAP", "Capability gap")}&nbsp; {c}', **H)


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
with tab_exec:
    st.markdown('<div class="sec-h">Processing record</div>', **H)
    st.dataframe(pd.DataFrame(state["execution_log"]), use_container_width=True,
                 hide_index=True)
    usage_events = state.get("model_usage") or []
    if usage_events:
        from services import costing
        cost = costing.session_cost(usage_events)
        scale = costing.at_scale(usage_events)

        st.markdown('<div class="sec-h">Measured cost of this session</div>', **H)
        st.markdown(ui.tiles([
            ("Model", cost["model"] or "—", f'{cost["provider"]} · {cost["calls"]} calls',
             ui.BRAND),
            ("Input tokens", f'{cost["input_tokens"]:,}', "measured", ui.MUTED),
            ("Output tokens", f'{cost["output_tokens"]:,}', "measured", ui.MUTED),
            ("Generation time", f'{cost["seconds"]:.0f}s', "wall clock", ui.MUTED),
            ("Local cost", f'₹{cost["local_inr"]:.3f}', "electricity only",
             ui.STATUS["SUPPORTED"]["fill"]),
            ("Hosted equivalent", f'₹{cost["api_equivalent_inr"]:.2f}',
             "same tokens, cloud API", ui.STATUS["PARTIAL"]["fill"]),
        ]), **H)
        st.markdown(
            f'<div class="note">Running the model locally costs '
            f'<b>₹{cost["local_inr"]:.3f}</b> per proposal in electricity and sends '
            f'nothing off the machine. The same workload on a hosted API would cost '
            f'<b>₹{cost["api_equivalent_inr"]:.2f}</b> — faster and higher quality, '
            f'but the tender and the firm\'s evidence would leave the tenant.</div>', **H)

        st.write("")
        st.markdown('<div class="sec-h">At 10,000 users · one proposal each per month</div>', **H)
        st.markdown(ui.tiles([
            ("Sessions", f'{scale["sessions"]:,}', "per month", ui.BRAND),
            ("Input tokens", f'{scale["input_tokens"]/1e6:.1f}M', "per month", ui.MUTED),
            ("Output tokens", f'{scale["output_tokens"]/1e6:.1f}M', "per month", ui.MUTED),
            ("Hosted API", f'₹{scale["api_monthly_inr"]:,.0f}', "per month",
             ui.STATUS["PARTIAL"]["fill"]),
            ("Self-hosted GPU", f'₹{scale["self_hosted_monthly_inr"]:,.0f}',
             f'{scale["self_hosted_gpu_hours"]:,.0f} GPU-hours', ui.STATUS["SUPPORTED"]["fill"]),
        ]), **H)
        st.markdown(
            '<div class="note">A single laptop cannot serve this concurrency, so at '
            'scale the choice is hosted inference or dedicated GPU capacity. Self-hosting '
            'keeps the offline guarantee and the cost becomes capacity, not tokens; the '
            'hosted route is cheaper to start and faster per request.</div>', **H)

        with st.expander("Assumptions behind these figures"):
            st.markdown('<div class="note">Token counts and elapsed time are measured. '
                        'Every rate below is an assumption — set them in config.py from '
                        'live pricing before quoting a number.</div>', **H)
            st.dataframe(pd.DataFrame(costing.assumptions(),
                                      columns=["Assumption", "Value"]),
                         use_container_width=True, hide_index=True)

        st.markdown('<div class="sec-h">Per-call record</div>', **H)
        st.dataframe(pd.DataFrame(usage_events), use_container_width=True, hide_index=True)
        st.download_button("Download model usage (CSV)",
                           pd.DataFrame(usage_events).to_csv(index=False),
                           file_name=f"model_usage_{state['run_id'][:8]}.csv",
                           mime="text/csv")
    else:
        st.markdown('<div class="sec-h">Model usage</div>', **H)
        st.markdown('<div class="note">This run used the deterministic offline '
                    'generator, so no model tokens were consumed. Run with '
                    '<code>LLM_PROVIDER=ollama</code> to record measured local usage '
                    'and cost.</div>', **H)
    if state.get("web_evidence"):
        st.markdown('<div class="sec-h">External context</div>', **H)
        st.markdown('<div class="note">Retained for background only. External sources '
                    'are never cited as evidence of the firm\'s experience.</div>', **H)
        st.dataframe(pd.DataFrame(state["web_evidence"]), use_container_width=True,
                     hide_index=True)
    try:
        trail = get_store().audit_trail(state["run_id"])
        if trail:
            with st.expander("Audit trail — every processing stage and human decision"):
                st.dataframe(pd.DataFrame(trail), use_container_width=True,
                             hide_index=True)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Review
# --------------------------------------------------------------------------- #
with tab_review:
    st.markdown(ui.verdict(
        export_ok,
        "Release gate open" if export_ok else "Release gate closed",
        "All sections have been approved and no unsubstantiated claims remain."
        if export_ok else "; ".join(export_reasons) + "."), **H)
    st.write("")

    st.markdown('<div class="sec-h">Commercial sign-off</div>', **H)
    price_approval = state.get("price_approval")
    price_current = review.commercial_approval_current(state)
    if price_approval and price_current:
        st.success("Partner approved commercial response: " +
                   price_approval["commercial_reference"])
    else:
        if price_approval and not price_current:
            # The approval record is kept for the audit trail, but it no longer
            # matches the content. Hiding the form here left the reviewer unable
            # to re-approve through the interface at all.
            st.warning(
                f"The commercial content changed after "
                f"{price_approval['reviewer']} approved "
                f"{price_approval['commercial_reference']}. That sign-off no "
                f"longer applies and release is blocked until it is given again.")
        st.caption("The partner approves the separate price or rate-card reference before release.")
        price_reviewer = st.text_input("Partner name", key="price_reviewer")
        sample_rate_ref = (state.get("system_inputs") or {}).get("rate_card", {}).get("reference", "")
        price_reference = st.text_input("Commercial response or rate-card reference",
                                        value=sample_rate_ref, key="price_reference")
        price_note = st.text_input("Approval note", key="price_note")
        if st.button("Approve commercial response"):
            try:
                review.approve_price(state, price_reviewer, price_reference, price_note)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    st.write("")

    st.markdown('<div class="sec-h">Section approval</div>', **H)
    for sec in draft.sections:
        approved = sec.review_status == ReviewDecision.APPROVED
        with st.expander(f"{sec.title}"
                         f"{'   ·   Approved' if approved else ''}"
                         f"{'   ·   Edited' if sec.human_edited else ''}"):
            st.markdown(
                f'<div class="pills">'
                f'{ui.pill("SUPPORTED" if approved else "NO_ROW", sec.review_status.value.replace("_", " ").title())}'
                f'{ui.pill("GAP", "Evidence gap") if sec.has_gaps else ""}</div>', **H)
            st.markdown(sec.content_markdown)
            comment = st.text_input("Reviewer comment", key=f"cm_{sec.section_id}")
            b1, b2, b3, b4 = st.columns(4)
            if b1.button("Approve", key=f"ap_{sec.section_id}", use_container_width=True):
                review.submit_section_decision(state, sec.section_id, "reviewer",
                                               ReviewDecision.APPROVED, comment)
                st.rerun()
            if b2.button("Request changes", key=f"cc_{sec.section_id}",
                         use_container_width=True):
                review.submit_section_decision(state, sec.section_id, "reviewer",
                                               ReviewDecision.CHANGES_REQUESTED, comment)
                st.rerun()
            if b3.button("Reject", key=f"rj_{sec.section_id}", use_container_width=True):
                review.submit_section_decision(state, sec.section_id, "reviewer",
                                               ReviewDecision.REJECTED, comment)
                st.rerun()
            if b4.button("Regenerate", key=f"rg_{sec.section_id}",
                         use_container_width=True):
                review.regenerate_section(state, sec.title)
                st.rerun()
            new_md = st.text_area(
                "Direct amendment — retained through subsequent regeneration of "
                "other sections",
                value=sec.content_markdown, key=f"ed_{sec.section_id}", height=140)
            if st.button("Save amendment", key=f"sv_{sec.section_id}"):
                review.apply_human_edit(state, sec.title, new_md)
                st.rerun()

    st.divider()
    st.markdown('<div class="sec-h">Unsubstantiated claims</div>', **H)
    gaps = review.unresolved_gaps(state)
    if not gaps:
        st.markdown(ui.verdict(True, "None outstanding",
                               "No unsubstantiated claims remain on this run."), **H)
    else:
        st.markdown('<div class="note">Each of the following blocks release until it '
                    'is resolved or overridden with a recorded justification.</div>', **H)
    for g in gaps:
        st.markdown(f'{ui.pill("GAP")}&nbsp; <b>{g.rfp_requirement[:80]}</b><br>'
                    f'<span style="color:{ui.MUTED}">{g.claim_text[:90]}</span>', **H)
        r = st.text_input(
            "Justification", key=f"ov_{g.trace_id}",
            placeholder="For example: no comparable engagement exceeds 30 percent; "
                        "the claim will be withdrawn from the submission.")
        if st.button("Record override", key=f"ovb_{g.trace_id}"):
            if r.strip():
                review.override_gap(state, g.trace_id, r)
                st.rerun()
            else:
                st.error("A justification is required.")

    st.divider()
    if export_ok:
        if st.button("Finalise proposal", type="primary"):
            review.finalize(state)
            st.rerun()
        d1, d2 = st.columns(2)
        d1.download_button("Download proposal (Markdown)",
                           export.render_markdown(state, enforce=False),
                           file_name=f"proposal_{state['run_id'][:8]}.md",
                           use_container_width=True)
        d2.download_button("Download traceability matrix (CSV)",
                           export.traceability_csv(state),
                           file_name=f"traceability_{state['run_id'][:8]}.csv",
                           use_container_width=True)
    else:
        st.markdown('<div class="note">Approve every section and resolve or override '
                    'each unsubstantiated claim to enable release.</div>', **H)
    st.write("")
    st.markdown('<div class="note">This system does not transmit or submit documents '
                'externally. Submission to the client remains a manual action '
                'performed outside the platform.</div>', **H)
