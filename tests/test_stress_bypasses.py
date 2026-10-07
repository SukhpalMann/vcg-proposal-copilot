"""Bypasses found by adversarial stress testing.

Every case here is a FALSE claim that reached SUPPORTED, or a stale document that
was used in silence. The only outcome this product cannot have is a false claim
presented as substantiated, so each one is pinned.
"""
from __future__ import annotations

import config
from models.schemas import AtomicClaim, EvidenceItem, VerificationStatus
from pipeline import conflicts
from pipeline.verification import numeric_consistency, verify_claim
from services.mock_llm import decompose_claims
from services.text import extract_numeric_tokens, significant_tokens
from services.vectorstore import VectorStore

# The real passage every numeric case is checked against.
TAT = (
    "The pilot was measured against a matched baseline of comparable branches over\n"
    "the preceding two quarters.\n\n"
    "- Pilot approval turnaround time was reduced by 18 percent versus the baseline.\n"
    "- Processing effort per application fell by 22 percent.\n"
    "- Manual handoffs per application dropped by 30 percent, from ten steps to seven.\n"
    "- Credit loss rate in the pilot cohort was unchanged within measurement error.\n"
)


# --------------------------------------------------------------------------- #
# Tokenisation: a sentence-final period rode along on the token, so "percent."
# never matched the generic-near-number set and became the single shared token
# that satisfied the distractor guard on its own.
# --------------------------------------------------------------------------- #
def test_sentence_punctuation_does_not_survive_tokenisation():
    assert significant_tokens("reduced by 18 percent versus the baseline.") == [
        "reduced", "percent", "versus", "baseline"]


def test_decimals_and_hyphenated_compounds_still_tokenise():
    assert "18.2" in significant_tokens("fell 18.2 percent")
    assert set(significant_tokens("a turnaround-time improvement")) >= {
        "turnaround", "time", "improvement"}


def test_a_figure_quoted_from_a_different_metric_is_rejected():
    """The passage reports 18 percent for TURNAROUND, and says the credit loss
    rate was unchanged. A claim of 18 percent about default rates passed because
    "percent." was the one token the two had in common."""
    assert numeric_consistency(
        "VCG reduced loan default rates by 18 percent.", TAT) is False
    assert numeric_consistency(
        "VCG reduced pilot approval turnaround time by 18 percent.", TAT) is True


# --------------------------------------------------------------------------- #
# A figure written in words produced no numeric token at all, so the numeric
# rule reported "not applicable" and the claim was judged on similarity alone.
# --------------------------------------------------------------------------- #
def test_numbers_written_in_words_are_tokenised():
    assert [t.value for t in extract_numeric_tokens("reduced by thirty-five percent")] == [35.0]
    assert [t.value for t in extract_numeric_tokens("reduced by eighteen percent")] == [18.0]
    assert [t.value for t in extract_numeric_tokens("a forty percent gain")] == [40.0]


def test_a_contradicted_figure_written_in_words_is_rejected():
    assert numeric_consistency(
        "VCG reduced turnaround time by thirty-five percent.", TAT) is False


def test_digitising_words_preserves_offsets_for_the_context_rule():
    """The context window and direction checks index into the original string."""
    text = "turnaround time was reduced by thirty-five percent versus baseline"
    tok = extract_numeric_tokens(text)[0]
    assert text[tok.start:tok.end].strip().endswith("percent")


# --------------------------------------------------------------------------- #
# Percentage points are not percent.
# --------------------------------------------------------------------------- #
def test_percentage_points_are_a_distinct_unit():
    pp = extract_numeric_tokens("a gain of 18 percentage points")[0]
    pct = extract_numeric_tokens("a gain of 18 percent")[0]
    assert pp.canonical_unit() == "percentage_points"
    assert pct.canonical_unit() == "percent"
    assert pp.canonical_unit() != pct.canonical_unit()


def test_a_percentage_point_claim_is_not_supported_by_a_percent_result():
    assert numeric_consistency(
        "VCG reduced turnaround time by 18 percentage points.", TAT) is False


# --------------------------------------------------------------------------- #
# A comparative qualifier makes the claim strictly stronger than the evidence.
# --------------------------------------------------------------------------- #
def test_more_than_is_not_established_by_exactly():
    for text in ("VCG reduced turnaround time by more than 18 percent.",
                 "VCG reduced turnaround time by over 18 percent.",
                 "VCG reduced turnaround time by at least 18 percent."):
        assert numeric_consistency(text, TAT) is False, text


def test_an_unqualified_claim_against_bounded_evidence_is_fine():
    ev = "Turnaround time was reduced by at least 18 percent versus the baseline."
    assert numeric_consistency(
        "VCG reduced turnaround time by 18 percent.", ev) is True


# --------------------------------------------------------------------------- #
# One passage cannot establish a property of every engagement.
# --------------------------------------------------------------------------- #
def _claim(sentence: str, evidence_id: str) -> AtomicClaim:
    raw = decompose_claims("Relevant Experience & Credentials",
                           f"{sentence} [[ev:{evidence_id}]]")
    assert len(raw) == 1, [r["claim_text"] for r in raw]
    return AtomicClaim(**raw[0])


def test_a_universal_quantifier_cannot_reach_supported():
    ev = EvidenceItem(
        evidence_id="CASE::00", source_id="CASE", chunk_id="CASE::00",
        title="CASE", chunk_text=TAT, category="CASE_STUDY", metadata={},
        source_path="data/corpus/CASE.md", relevance_score=0.9, selected=True,
    )
    store = VectorStore.ensure_seeded()
    sem = VectorStore.load().semantic_similarity

    true_one = _claim("VCG reduced pilot approval turnaround time by 18 percent.",
                      "CASE::00")
    assert verify_claim(true_one, {"CASE::00": ev}, sem)["status"] == \
        VerificationStatus.SUPPORTED

    generalised = _claim(
        "VCG has reduced pilot approval turnaround time by 18 percent for every "
        "lending client.", "CASE::00")
    result = verify_claim(generalised, {"CASE::00": ev}, sem)
    assert result["status"] == VerificationStatus.PARTIAL
    assert "generalises" in result["reason"]


# --------------------------------------------------------------------------- #
# A document that declares itself superseded was only flagged when its
# replacement happened to be retrieved too -- which top-k retrieval rarely does.
# --------------------------------------------------------------------------- #
def _ev(eid, src, text, **kw):
    return EvidenceItem(evidence_id=eid, source_id=src, chunk_id=eid, title=src,
                        chunk_text=text, category="CASE_STUDY", metadata={},
                        source_path=f"data/corpus/{src}.md",
                        relevance_score=0.9, selected=True, **kw)


def test_a_superseded_document_is_flagged_even_if_its_successor_is_absent():
    stale = _ev("OLD::00", "CASE_OLD", "Turnaround time was reduced by 40 percent.",
                superseded_by="CASE_NEW")
    state = {"selected_evidence": {"CHK-001": [stale]}, "errors": [],
             "warnings": [], "execution_log": []}
    found = conflicts.run(state)["evidence_conflicts"]
    assert [c.conflict_type for c in found] == ["superseded"]
    assert "NOT retrieved" in found[0].description


def test_a_document_marked_none_is_not_flagged():
    fine = _ev("OK::00", "CASE_OK", "Turnaround time fell 18 percent.",
               superseded_by="none")
    state = {"selected_evidence": {"CHK-001": [fine]}, "errors": [],
             "warnings": [], "execution_log": []}
    assert conflicts.run(state)["evidence_conflicts"] == []


# --------------------------------------------------------------------------- #
# Found by auditing a recorded run's SUPPORTED claims against their sources.
# A drafted sentence ends with its citation tag, which sat between the full stop
# and the next capital, so the sentence boundary was never recognised: three
# sentences became one 402-character "atomic" claim and verifying any part of it
# stamped the whole blob SUPPORTED.
# --------------------------------------------------------------------------- #
CITED_BLOCK = (
    "VCG's team brings significant experience in redesigning retail lending "
    "operations for Indian banks. [[ev:CHK-014::CV_001::00.00]] Ananya Mehta, "
    "Partner at VCG, possesses 18 years of experience in banking and lending "
    "operations. [[ev:CHK-014::CASE_BANK_001::00.00]] Furthermore, a key team "
    "member has previously led retail lending operations redesign for a large "
    "Indian bank. [[ev:CHK-014::CASE_BANK_001::02.00]]"
)


def test_a_citation_tag_does_not_hide_a_sentence_boundary():
    from services.llm import sentences

    assert len(sentences(CITED_BLOCK)) == 3


def test_each_sentence_keeps_its_own_citation():
    claims = decompose_claims("Team & Credentials", CITED_BLOCK)
    assert len(claims) == 3, [c["claim_text"] for c in claims]
    assert [c["cited_evidence_ids"] for c in claims] == [
        ["CHK-014::CV_001::00.00"],
        ["CHK-014::CASE_BANK_001::00.00"],
        ["CHK-014::CASE_BANK_001::02.00"],
    ]
    assert all(len(c["claim_text"]) < 200 for c in claims)


def test_plain_prose_still_splits_and_a_decimal_does_not():
    from services.llm import sentences

    assert len(sentences("We did one thing. Then we did another.")) == 2
    assert len(sentences("Turnaround fell 18.2 percent against baseline.")) == 1


# --------------------------------------------------------------------------- #
# The headline substantiation figures counted traceability ENTRIES, not claims,
# so a claim linked to two requirements was counted twice.
# --------------------------------------------------------------------------- #
def test_substantiation_counts_distinct_claims_not_trace_entries():
    from pipeline.graph import continue_approved_pipeline, run_pipeline
    from pipeline.qualification import record_decision

    config.ensure_dirs()
    VectorStore.ensure_seeded()
    state = run_pipeline(
        str(config.FIXTURE_DIR / "abc_bank_lending_transformation.md"),
        run_id="test-counts", persist=False,
    )
    record_decision(state, "BID", "test", "count regression")
    state = continue_approved_pipeline(state, persist=False)

    draft = state["proposal_draft"]
    results = state["_verification_results"]
    linked = {e.claim_id for e in draft.overall_traceability if e.claim_id != "(none)"}

    def claims_with(status: str) -> int:
        return len({cid for cid in linked
                    if results[cid]["status"].value == status})

    assert draft.supported_claim_count == claims_with("SUPPORTED")
    assert draft.partial_claim_count == claims_with("PARTIAL")
    assert draft.gap_claim_count == claims_with("GAP")

    # and a claim linked twice must not inflate the count
    entries = [e for e in draft.overall_traceability if e.claim_id != "(none)"]
    assert len(entries) >= len({e.claim_id for e in entries})


# --------------------------------------------------------------------------- #
# The generating model does not keep the tag syntax clean. gemma3 emits
#   [[ev:A] and [ev:B]]
# which the tag stripper took as two tags, leaving " and " in the prose. That
# residue stood between a full stop and the next capital, so the boundary was
# missed and the cited tenure fact carried an uncited assertion to SUPPORTED.
# --------------------------------------------------------------------------- #
MALFORMED = (
    "Ananya Mehta, Partner at VCG, possesses 18 years of experience in banking "
    "and lending operations. [[ev:CHK-014::CV_001::00.00] and "
    "[ev:CHK-014::CASE_BANK_001::00.00]]  Furthermore, a key team member has "
    "previously led retail lending operations redesign for a large Indian bank. "
    "[[ev:CHK-003::CV_001::01.00]]"
)


def test_malformed_tag_runs_are_repaired():
    from services.mock_llm import normalise_tags

    out = normalise_tags(MALFORMED)
    assert "[[ev:CHK-014::CV_001::00.00]] [[ev:CHK-014::CASE_BANK_001::00.00]]" in out
    assert "] and [" not in out


def test_a_malformed_tag_run_does_not_merge_two_sentences():
    claims = decompose_claims("Team & Credentials", MALFORMED)
    assert len(claims) == 2, [c["claim_text"] for c in claims]
    assert " and  Furthermore" not in claims[0]["claim_text"]
    assert claims[0]["cited_evidence_ids"] == [
        "CHK-014::CV_001::00.00", "CHK-014::CASE_BANK_001::00.00"]
    assert claims[1]["cited_evidence_ids"] == ["CHK-003::CV_001::01.00"]


def test_longer_connector_runs_collapse():
    from services.mock_llm import normalise_tags

    out = normalise_tags("x. [[ev:A] and [ev:B] and [ev:C]]")
    assert out.count("[[ev:") == 3
    assert " and " not in out


# --------------------------------------------------------------------------- #
# A claim citing several passages was verified against the first that resolved,
# so a contradiction in the second or third was never seen -- and Rule C is
# meant to be unoverridable.
# --------------------------------------------------------------------------- #
def test_a_contradiction_in_any_cited_passage_is_a_gap():
    clean = EvidenceItem(
        evidence_id="CV::00", source_id="CV", chunk_id="CV::00", title="CV",
        chunk_text="Ananya Mehta is a Partner at VCG with 18 years of experience "
                   "in banking and lending operations.",
        category="TEAM_CV", metadata={}, source_path="data/corpus/CV.md",
        relevance_score=0.9, selected=True,
    )
    contradicting = EvidenceItem(
        evidence_id="CASE::00", source_id="CASE", chunk_id="CASE::00", title="CASE",
        chunk_text="Ananya Mehta has 11 years of experience in banking and lending "
                   "operations at the firm.",
        category="CASE_STUDY", metadata={}, source_path="data/corpus/CASE.md",
        relevance_score=0.8, selected=True,
    )
    sem = VectorStore.load().semantic_similarity
    sentence = ("Ananya Mehta, Partner, has 18 years of experience in banking and "
                "lending operations.")

    # cited alone, the clean passage supports it
    only_clean = _claim(sentence, "CV::00")
    assert verify_claim(only_clean, {"CV::00": clean}, sem)["status"] == \
        VerificationStatus.SUPPORTED

    # the clean passage FIRST, the contradicting one second: still a gap
    raw = decompose_claims("Team & Credentials",
                           f"{sentence} [[ev:CV::00]] [[ev:CASE::00]]")
    assert len(raw) == 1
    both = AtomicClaim(**raw[0])
    assert both.cited_evidence_ids == ["CV::00", "CASE::00"]
    result = verify_claim(both, {"CV::00": clean, "CASE::00": contradicting}, sem)
    assert result["status"] == VerificationStatus.GAP
    assert "CONTRADICTED" in result["reason"]


# --------------------------------------------------------------------------- #
# Checking every cited passage and then taking the best status made Rule E
# overridable by which passage scored highest. "mirroring successful engagements
# for other Indian banks", citing one Indian engagement and one Southeast Asian
# one, was SUPPORTED by the Indian passage while the other flagged the geography
# mismatch that makes the plural false. A disqualifying finding against ANY cited
# passage applies.
# --------------------------------------------------------------------------- #
def test_a_geography_mismatch_in_any_cited_passage_blocks_supported():
    india = EvidenceItem(
        evidence_id="IN::00", source_id="CASE_IN", chunk_id="IN::00", title="CASE_IN",
        chunk_text="A large Indian retail bank engaged VCG to redesign its consumer "
                   "lending operations across personal loans and two-wheeler finance.",
        category="CASE_STUDY", metadata={"region": "India", "industry": "banking"},
        source_path="data/corpus/CASE_IN.md", relevance_score=0.9, selected=True,
    )
    sea = EvidenceItem(
        evidence_id="SEA::00", source_id="CASE_SEA", chunk_id="SEA::00", title="CASE_SEA",
        chunk_text="A mid-size Southeast Asian bank asked VCG to improve its small and "
                   "medium enterprise lending process.",
        category="CASE_STUDY",
        metadata={"region": "Southeast Asia", "industry": "banking"},
        source_path="data/corpus/CASE_SEA.md", relevance_score=0.8, selected=True,
    )
    sem = VectorStore.load().semantic_similarity
    sentence = ("VCG's approach to redesigning consumer lending operations centers on a "
                "time-boxed pilot, mirroring successful engagements for other Indian banks.")

    # the Indian passage alone supports it
    alone = _claim(sentence, "IN::00")
    assert verify_claim(alone, {"IN::00": india}, sem)["status"] == \
        VerificationStatus.SUPPORTED

    # citing the Southeast Asian engagement as one of those "Indian banks" must not
    # be rescued by the Indian passage scoring better
    raw = decompose_claims("Approach",
                           f"{sentence} [[ev:IN::00]] [[ev:SEA::00]]")
    assert len(raw) == 1
    both = AtomicClaim(**raw[0])
    assert both.cited_evidence_ids == ["IN::00", "SEA::00"]
    result = verify_claim(both, {"IN::00": india, "SEA::00": sea}, sem)
    assert result["status"] == VerificationStatus.PARTIAL
    assert "geography" in result["reason"]


# --------------------------------------------------------------------------- #
# "Every citation resolves to selected evidence" was enforced and reported for
# forward-looking claims only. On a verifiable claim the unresolvable ids were
# filtered out in silence: a claim citing one real passage and one invented id
# came back SUPPORTED, reported nothing, and left the invented id on the page.
# --------------------------------------------------------------------------- #
CV = EvidenceItem(
    evidence_id="CV::00", source_id="CV", chunk_id="CV::00", title="CV",
    chunk_text="Ananya Mehta is a Partner at VCG with 18 years of experience in "
               "banking and lending operations.",
    category="TEAM_CV", metadata={}, source_path="data/corpus/CV.md",
    relevance_score=0.9, selected=True,
)
TENURE = ("Ananya Mehta, Partner, has 18 years of experience in banking and "
          "lending operations.")


def test_a_fabricated_citation_is_reported_and_blocks_supported():
    sem = VectorStore.load().semantic_similarity

    clean = AtomicClaim(**decompose_claims("Team", f"{TENURE} [[ev:CV::00]]")[0])
    good = verify_claim(clean, {"CV::00": CV}, sem)
    assert good["status"] == VerificationStatus.SUPPORTED
    assert good["unresolved_citations"] == []
    assert good["citation_valid"] is True

    invented = AtomicClaim(**decompose_claims(
        "Team", f"{TENURE} [[ev:CV::00]] [[ev:TOTALLY_MADE_UP::99]]")[0])
    assert invented.cited_evidence_ids == ["CV::00", "TOTALLY_MADE_UP::99"]
    bad = verify_claim(invented, {"CV::00": CV}, sem)
    assert bad["status"] == VerificationStatus.PARTIAL
    assert bad["unresolved_citations"] == ["TOTALLY_MADE_UP::99"]
    assert bad["citation_valid"] is False
    assert "CITATION REJECTED" in bad["reason"]


def test_a_rejected_citation_is_actually_removed_from_the_drafted_text():
    """The verdict said a rejected citation "was stripped" while the fabricated id
    stayed on the page, so the reader saw provenance the verifier had rejected."""
    from pipeline.graph import continue_approved_pipeline, run_pipeline
    from pipeline.qualification import record_decision

    config.ensure_dirs()
    VectorStore.ensure_seeded()
    state = run_pipeline(
        str(config.FIXTURE_DIR / "abc_bank_lending_transformation.md"),
        run_id="test-stripped", persist=False,
    )
    record_decision(state, "BID", "test", "citation stripping")
    state = continue_approved_pipeline(state, persist=False)

    rejected = {cid for r in state["_verification_results"].values()
                for cid in (r.get("unresolved_citations") or [])}
    body = "\n".join(s.content_markdown for s in state["proposal_draft"].sections)
    for cid in rejected:
        assert f"[[ev:{cid}]]" not in body, cid


# --------------------------------------------------------------------------- #
# The model writes the id inside angle brackets, the way the prompt's own
# placeholder shows it: [[ev:<CHK-014::CV_001::00.00>]]. "<" is not in the id
# character class, so the citation was not extracted at all -- a VALID citation
# was discarded, its claim became an orphan GAP, and because no claim owned the
# tag nothing reported it, so it stayed on the page as unchecked provenance.
# --------------------------------------------------------------------------- #
def test_angle_bracketed_ids_are_unwrapped():
    from services.mock_llm import normalise_tags

    assert normalise_tags("x [[ev:<CHK-003::CASE_BANK_001::01.00>]].") == \
        "x [[ev:CHK-003::CASE_BANK_001::01.00]]."
    assert normalise_tags("y [[req:<CHK-018>]].") == "y [[req:CHK-018]]."


def test_an_angle_bracketed_citation_is_extracted_not_discarded():
    claims = decompose_claims(
        "Approach",
        "Ananya Mehta is a Partner at VCG with 18 years of experience "
        "[[ev:<CHK-014::CV_001::00.00>]].")
    assert len(claims) == 1
    assert claims[0]["cited_evidence_ids"] == ["CHK-014::CV_001::00.00"]


def test_the_rendered_draft_never_shows_a_citation_that_does_not_resolve():
    """Keyed on what resolves, not only on what a claim reported as unresolved: a
    tag the extractor could not parse belonged to no claim, so nothing reported it."""
    from pipeline.graph import continue_approved_pipeline, run_pipeline
    from pipeline.qualification import record_decision

    config.ensure_dirs()
    VectorStore.ensure_seeded()
    state = run_pipeline(
        str(config.FIXTURE_DIR / "abc_bank_lending_transformation.md"),
        run_id="test-citations-shown", persist=False,
    )
    record_decision(state, "BID", "test", "citation display")
    state = continue_approved_pipeline(state, persist=False)

    resolvable = set()
    for evs in state["selected_evidence"].values():
        for ev in evs:
            resolvable.update((ev.evidence_id, ev.source_id, ev.chunk_id))

    import re as _re
    body = "\n".join(s.content_markdown for s in state["proposal_draft"].sections)
    shown = set(_re.findall(r"\[\[ev:([^\]]+)\]\]", body))
    assert shown, "no citations rendered at all"
    assert not (shown - resolvable), sorted(shown - resolvable)


# --------------------------------------------------------------------------- #
# Third bypass in a row from an allow-list of verbs. "involved a tiered
# auto-decisioning model ... resulting in a 18 percent reduction in approval
# turnaround time versus a matched baseline" is a stated past result, and
# "involved"/"resulting" were on neither the achievement nor the past-result
# list, so the figure was exempted and reported as a prospective statement.
# --------------------------------------------------------------------------- #
def test_any_figure_is_checked_whatever_the_verb():
    from services.mock_llm import is_verifiable_assertion
    from services.text import extract_named_entities, extract_numeric_tokens

    def verifiable(text: str) -> bool:
        return is_verifiable_assertion(
            text, [t.raw for t in extract_numeric_tokens(text)],
            extract_named_entities(text))

    assert verifiable(
        "This pilot, mirroring the approach used for a large Indian retail bank, "
        "involved a tiered auto-decisioning model, resulting in a 18 percent "
        "reduction in approval turnaround time versus a matched baseline.")
    assert verifiable("The engagement wrapped up inside 9 weeks.")
    assert verifiable("A comparable programme put Rs 240 crore back on the table.")
    # still not claims
    assert not verifiable("We will deliver the pilot in weeks 1 to 4.")
    assert not verifiable("This proposal is structured in 7 sections as set out below.")


def test_the_pronoun_one_is_not_a_figure():
    """Digitising word-numbers turned "the result is one the client can audit"
    into a 1, which made a methodology sentence a numeric claim."""
    assert extract_numeric_tokens("is one the client can audit") == []
    assert [t.value for t in extract_numeric_tokens("twenty-one percent")] == [21.0]


# --------------------------------------------------------------------------- #
# The model also makes the citation the sentence's subject, so stripping the tag
# left a sentence starting with a lowercase verb; the capital-letter requirement
# never fired and four assertions became one 650-character claim with one verdict.
# --------------------------------------------------------------------------- #
SUBJECT_TAGS = (
    "VCG has a proven track record of redesigning retail lending operations for "
    "banks across India. [[ev:CHK-014::CASE_BANK_001::00.00]] demonstrates our "
    "experience, with Ananya Mehta leading the practice. "
    "[[ev:CHK-003::CV_001::01.00]] details a previous engagement where VCG mapped "
    "the value stream. [[ev:CHK-016::CASE_BANK_001::02.00]] highlights a pilot "
    "where approval turnaround time was reduced by 18 percent."
)


def test_a_citation_acting_as_the_subject_still_starts_a_new_sentence():
    from services.llm import sentences

    out = sentences(SUBJECT_TAGS)
    assert len(out) == 4, out
    assert all(len(s) < 220 for s in out), [len(s) for s in out]


def test_a_trailing_citation_still_stays_with_its_own_sentence():
    from services.llm import sentences

    assert sentences("We did one thing. [[ev:A]] Then we did another. [[ev:B]]") == [
        "We did one thing. [[ev:A]]", "Then we did another. [[ev:B]]"]


def test_bare_tag_without_kind_is_repaired_not_dropped():
    """Observed live (gpt-oss-20b, v2 prompt): the "ev:" prefix was dropped, so a
    sentence citing real evidence parsed as an uncited orphan."""
    from services.mock_llm import decompose_claims, normalise_tags

    live = ("This inefficiency has caused the bank to lose volume to digital-first "
            "lenders [[CHK-014::CASE_BANK_001::00.00]].")
    assert "[[ev:CHK-014::CASE_BANK_001::00.00]]" in normalise_tags(live)
    assert normalise_tags("x [[CHK-014]].") == "x [[req:CHK-014]]."
    assert normalise_tags("x [[CV_001]].") == "x [[ev:CV_001]]."
    # anything that does not look like an id is left alone
    assert normalise_tags("[[EVIDENCE GAP: none]]") == "[[EVIDENCE GAP: none]]"

    claims = decompose_claims("Context", "VCG reduced turnaround time by 18 percent "
                                        "[[CHK-003::CASE_BANK_001::01.00]].")
    assert claims and claims[0]["cited_evidence_ids"] == ["CHK-003::CASE_BANK_001::01.00"]
