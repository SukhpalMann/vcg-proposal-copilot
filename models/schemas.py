"""Pydantic v2 data models. Mirrors SPEC.md Section 7 with a small number of
helper methods and one extra model (EvidenceMatch) the deterministic verifier
uses as its per-claim working record.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class EvidenceCategory(str, Enum):
    CASE_STUDY = "CASE_STUDY"
    METHODOLOGY = "METHODOLOGY"
    TEAM_CV = "TEAM_CV"
    WINNING_PROPOSAL = "WINNING_PROPOSAL"
    MARKET_RESEARCH = "MARKET_RESEARCH"
    OTHER = "OTHER"


class RequirementCategory(str, Enum):
    CONTENT = "CONTENT"
    PROCEDURAL = "PROCEDURAL"
    COMMERCIAL = "COMMERCIAL"
    COMPLIANCE = "COMPLIANCE"


class RequirementHandling(str, Enum):
    NEEDS_EVIDENCE = "NEEDS_EVIDENCE"
    NEEDS_HUMAN_INPUT = "NEEDS_HUMAN_INPUT"
    TEMPLATE_SATISFIABLE = "TEMPLATE_SATISFIABLE"
    PROCEDURAL_ONLY = "PROCEDURAL_ONLY"
    CAPABILITY_GAP = "CAPABILITY_GAP"


class VerificationStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    GAP = "GAP"
    FORWARD_LOOKING = "FORWARD_LOOKING"
    # Requirement rows with no drafted claim. These were never verified, so they
    # must not be reported as PARTIAL (which means "checked, and half-holds").
    NARRATIVE = "NARRATIVE"          # answered in approach/workplan prose
    HUMAN_INPUT = "HUMAN_INPUT"      # awaits partner-supplied content


class ReviewDecision(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    CHANGES_REQUESTED = "CHANGES_REQUESTED"


class ClaimType(str, Enum):
    FACTUAL = "FACTUAL"
    NUMERIC = "NUMERIC"
    EXPERIENCE = "EXPERIENCE"
    METHODOLOGY = "METHODOLOGY"
    PROPOSED_ACTION = "PROPOSED_ACTION"
    OPINION = "OPINION"


# --------------------------------------------------------------------------- #
# Core models
# --------------------------------------------------------------------------- #
class SourceSpan(BaseModel):
    page: int | None = None
    section: str | None = None
    quote: str                       # must be locatable in normalized RFP text (SPEC Section 8)
    start_char: int | None = None
    end_char: int | None = None


class RFPRequirement(BaseModel):
    requirement_id: str
    text: str
    category: RequirementCategory
    handling: RequirementHandling
    mandatory: bool | None = None
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    source_span: SourceSpan
    valid: bool = True               # False => ungrounded, do not pass downstream
    validation_note: str | None = None


class RFPRequirements(BaseModel):
    """Structured extraction result for one RFP."""

    client: str | None = None
    problem_statement: str | None = None
    timeline: str | None = None
    scope_items: list[str] = Field(default_factory=list)
    deliverables: list[str] = Field(default_factory=list)
    evaluation_criteria: list[str] = Field(default_factory=list)
    requirements: list[RFPRequirement] = Field(default_factory=list)
    procedural_checklist: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class EvidenceChecklistItem(BaseModel):
    checklist_id: str
    requirement_id: str
    requirement_text: str
    evidence_need: str
    category: RequirementCategory
    handling: RequirementHandling
    target_section: str
    satisfied: bool = False
    status: str = "PENDING"          # PENDING | COVERED | PARTIAL | GAP


class EvidenceItem(BaseModel):
    evidence_id: str
    source_id: str
    title: str
    category: EvidenceCategory
    chunk_id: str
    chunk_text: str                  # INVARIANT: verbatim source text, never an LLM summary
    source_path: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    metadata_match_score: float = 0.0
    reasoning: str = ""
    selected: bool = False
    rejection_reason: str | None = None
    usage_restriction: str = "internal_only"     # or "nda_restricted"
    freshness_date: str | None = None
    superseded_by: str | None = None
    conflict_flag: bool = False
    metadata: dict = Field(default_factory=dict)


class EvidenceConflict(BaseModel):
    conflict_id: str
    evidence_ids: list[str]
    conflict_type: str               # numeric_mismatch | attribution_mismatch | superseded
    description: str
    requires_human_resolution: bool = True


class AtomicClaim(BaseModel):
    claim_id: str
    section_name: str
    claim_text: str
    claim_type: ClaimType
    source_checklist_id: str | None = None   # which checklist item this claim was drafted for
    cited_evidence_ids: list[str] = Field(default_factory=list)
    numeric_tokens: list[str] = Field(default_factory=list)
    named_entities: list[str] = Field(default_factory=list)      # empty => no attribution check
    context_qualifiers: list[str] = Field(default_factory=list)  # e.g. ["India"], ["SME"]
    requires_verification: bool = True


class EvidenceMatch(BaseModel):
    """The verifier's working record for one (claim, best-evidence) pair.

    Attribute names match the decision-rule pseudocode in SPEC Section 15.
    """

    claim_id: str
    evidence_id: str | None = None
    chunk_text: str | None = None
    semantic_similarity: float = 0.0
    lexical_overlap: float = 0.0
    numeric_match: bool | None = None        # None => no numeric token in claim
    attribution_valid: bool = True           # DEFAULT TRUE -- only False on a real mismatch
    context_mismatch: bool = False
    conflict_flag: bool = False
    evidence_metadata: dict = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    def metadata_contradicts(self, qualifier: str) -> bool:
        """True only if the matched evidence's own metadata *contradicts* the
        qualifier (not merely fails to mention it).

        The groups come from services.text, the same vocabulary the extractor
        uses. They used to be a private copy here that knew four geographic
        groups while the extractor knew ten strings, so a qualifier this method
        could have contradicted was often never extracted in the first place.
        """
        from services.text import GEO_GROUPS, IND_GROUPS, qualifier_group

        q = qualifier.strip().lower()
        if not q:
            return False
        region = str(self.evidence_metadata.get("region", "")).lower()
        industry = str(self.evidence_metadata.get("industry", "")).lower()
        subsector = str(self.evidence_metadata.get("subsector", "")).lower()
        haystack = " ".join([region, industry, subsector]).replace("_", " ")
        if not haystack.strip():
            return False                      # no metadata => cannot contradict
        # "global" evidence is not tied to a region, so it contradicts none.
        for groups, hay in ((GEO_GROUPS, region.replace("_", " ")),
                            (IND_GROUPS, haystack)):
            if groups is GEO_GROUPS and hay.strip() in ("", "global"):
                continue
            q_group = qualifier_group(q, groups)
            if q_group is None:
                continue
            hay_group = qualifier_group(hay, groups)
            if hay_group is not None and hay_group is not q_group:
                return True
        return False


class TraceabilityEntry(BaseModel):
    trace_id: str
    requirement_id: str
    rfp_requirement: str
    requirement_source_span: SourceSpan | None = None
    matched_evidence_id: str | None = None
    matched_chunk_text: str | None = None
    draft_section: str
    claim_id: str
    claim_text: str
    verification_status: VerificationStatus
    confidence_score: float = Field(ge=0.0, le=1.0)
    lexical_overlap: float = 0.0
    semantic_similarity: float = 0.0
    numeric_match: bool | None = None        # None => not applicable
    attribution_valid: bool = True           # DEFAULT TRUE (SPEC Section 11 fix)
    conflict_flag: bool = False
    verification_reason: str = ""
    reviewer_decision: ReviewDecision = ReviewDecision.PENDING
    reviewer_comment: str | None = None


class ProposalSection(BaseModel):
    section_id: str
    title: str
    content_markdown: str
    version: int = 1
    evidence_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    has_gaps: bool = False
    review_status: ReviewDecision = ReviewDecision.PENDING
    human_edited: bool = False


class ProposalDraft(BaseModel):
    proposal_id: str
    client: str
    rfp_id: str
    run_id: str
    sections: list[ProposalSection] = Field(default_factory=list)
    overall_traceability: list[TraceabilityEntry] = Field(default_factory=list)
    supported_claim_count: int = 0
    partial_claim_count: int = 0
    gap_claim_count: int = 0
    status: ReviewDecision = ReviewDecision.PENDING


class ReviewDecisionRecord(BaseModel):
    decision_id: str
    run_id: str
    section_id: str
    reviewer: str
    decision: ReviewDecision
    comment: str | None = None
    timestamp: datetime


class AuditLogEntry(BaseModel):
    event_id: str
    run_id: str
    stage: str
    actor: str                       # "agent" | "human"
    timestamp: datetime
    status: str
    notes: str | None = None
