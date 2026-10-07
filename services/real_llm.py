"""Real LLM implementations of the four delegated operations (SPEC Section 5).

Routed through LiteLLM when ``LLM_PROVIDER=litellm``; the model id comes from
``LLM_MODEL`` in .env and is never hardcoded here. The mock provider
(services/mock_llm.py) remains the default so the demo + tests run offline.

Design choices that keep the rest of the pipeline unchanged:
  * drafting must emit the same citation convention the deterministic verifier
    depends on -- ``[[ev:<evidence_id>]]`` for a cited chunk and
    ``[[req:<checklist_id>]]`` for the requirement a claim answers;
  * claim decomposition asks the model only to *split* prose into atomic
    sentences -- the numeric/entity/qualifier/tag extraction is then done by the
    same deterministic helpers the mock uses, so verification behaviour is
    identical regardless of provider.

Every model call uses temperature 0 and is defensively parsed; on any parse
failure the caller gets a clear RuntimeError rather than a silent bad result.
"""
from __future__ import annotations

import config

import json
import re

from services import mock_llm
from services.llm import parse_sections, sentences, split_compound
from services.text import (
    extract_context_qualifiers,
    extract_named_entities,
    extract_numeric_tokens,
)

_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.S)


def _json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(text)
        if not m:
            raise RuntimeError(f"LLM did not return JSON:\n{text[:500]}")
        return json.loads(m.group(0))


# --------------------------------------------------------------------------- #
# 1. Requirement extraction
# --------------------------------------------------------------------------- #
_EXTRACT_SYS = """You extract structured requirements from a consulting RFP.
Return ONLY JSON, no prose. Schema:
{
 "client": str|null, "problem_statement": str|null, "timeline": str|null,
 "scope_items": [str], "deliverables": [str], "evaluation_criteria": [str],
 "requirements": [{
   "text": str,                # one atomic requirement -- SPLIT compound bullets
   "category": "CONTENT"|"PROCEDURAL"|"COMMERCIAL"|"COMPLIANCE",
   "handling": "NEEDS_EVIDENCE"|"NEEDS_HUMAN_INPUT"|"TEMPLATE_SATISFIABLE"|"PROCEDURAL_ONLY"|"CAPABILITY_GAP",
   "mandatory": bool|null,
   "extraction_confidence": number,   # 0..1
   "quote": str,               # VERBATIM substring of the RFP that this came from
   "section": str              # target proposal section
 }],
 "procedural_checklist": [str], # NDA / references / format / submission-channel items, verbatim
 "warnings": [str]
}
Rules: quote MUST be copied verbatim from the RFP text (used for an anti-hallucination
check). Put NDA/reference/formatting/submission items in procedural_checklist AND as
PROCEDURAL_ONLY requirements. If a mandatory requirement asks for a capability with no
plausible consulting-firm coverage, mark it CAPABILITY_GAP. Never invent evaluation
criteria that are not in the RFP -- add a warning instead.
Category and handling rules: work scope, deliverables, timelines, and proposed
methodology are CONTENT and TEMPLATE_SATISFIABLE unless the RFP explicitly asks
for proof of past firm experience. PROCEDURAL_ONLY means submission channel,
signature, document format, annexure, or deadline. Past projects, certifications,
and credentials are CONTENT or COMPLIANCE with NEEDS_EVIDENCE. Do not call a
scope-of-work activity a procedural submission rule."""


def extract_requirements(rfp_text: str, filename: str = "", *, complete=None) -> dict:
    raw = complete(_EXTRACT_SYS, f"RFP:\n\n{rfp_text}")
    data = _json(raw)
    rid = 0
    for r in data.get("requirements", []):
        rid += 1
        r.setdefault("requirement_id", f"REQ-{rid:03d}")
        r.setdefault("mandatory", None)
        r.setdefault("extraction_confidence", 0.7)
        r.setdefault("section", "Context & Problem Understanding")
    data.setdefault("scope_items", parse_sections(rfp_text).get("scope of work", "") and [])
    for key in ("scope_items", "deliverables", "evaluation_criteria",
                "procedural_checklist", "warnings"):
        data.setdefault(key, [])
    return data


# --------------------------------------------------------------------------- #
# 2. Response planning
# --------------------------------------------------------------------------- #
_PLAN_SYS = """You turn validated RFP requirements into a proposal plan.
Return ONLY JSON:
{
 "checklist": [{"requirement_id": str, "evidence_need": str, "target_section": str}],
 "human_input_requirements": [str]
}
evidence_need = one sentence describing what internal evidence would satisfy the
requirement. Skip PROCEDURAL_ONLY requirements. Route NEEDS_HUMAN_INPUT ones into
human_input_requirements as well."""


def plan_response(rfp_data: dict, *, complete=None) -> dict:
    reqs = rfp_data["requirements"]
    raw = complete(_PLAN_SYS, json.dumps({"requirements": reqs}))
    plan = _json(raw)
    by_id = {r["requirement_id"]: r for r in reqs}
    checklist = []
    for i, entry in enumerate(plan.get("checklist", []), 1):
        req = by_id.get(entry["requirement_id"])
        if not req or req["handling"] == "PROCEDURAL_ONLY":
            continue
        checklist.append({
            "checklist_id": f"CHK-{i:03d}",
            "requirement_id": req["requirement_id"],
            "requirement_text": req["text"],
            "evidence_need": entry.get("evidence_need",
                                       f"Evidence for: {req['text']}"),
            "category": req["category"],
            "handling": req["handling"],
            "target_section": entry.get("target_section", req.get("section",
                                        "Context & Problem Understanding")),
            "status": "PENDING",
        })
    return {
        "checklist": checklist,
        "proposal_outline": mock_llm.STANDARD_OUTLINE,
        "human_input_requirements": plan.get("human_input_requirements", []),
    }


# --------------------------------------------------------------------------- #
# 3. Drafting
# --------------------------------------------------------------------------- #
_DRAFT_SYS = """You draft ONE section of a consulting proposal from supplied evidence only.
Output Markdown prose (no JSON).

HARD RULES:
- Factual statements about our firm may ONLY use the supplied evidence chunks.
- Immediately after a sentence built from a chunk, cite it: [[ev:<evidence_id>]].
- If a sentence answers a specific checklist item, also tag: [[req:<checklist_id>]].
- Never invent client names, numbers, percentages, credentials, or durations.
- Proposed FUTURE actions use prospective phrasing ("We propose...", "In weeks 1-3
  the team will...") and carry NO evidence citation.
- If evidence is insufficient for a checklist item, write exactly:
  [EVIDENCE GAP: <what is missing>]
- For pricing / named staffing / commercial terms write exactly:
  **HUMAN INPUT REQUIRED:** <what the engagement partner must supply>
Keep it tight: 2-5 sentences per checklist item."""

# v2: written after observing the local model invent evidence ids and restate
# figures loosely. Three changes, each aimed at one measured failure:
#   1. the exact list of citable ids is given, and anything else is forbidden;
#   2. figures must be copied verbatim from the passage being cited;
#   3. a tender threshold the evidence does not reach must become a gap marker,
#      not a rounded-up claim.
# scripts/prompt_ab.py measures v1 against v2; DRAFT_PROMPT_VERSION selects one.
_DRAFT_SYS_V2 = """You draft ONE section of a consulting proposal from supplied evidence only.
Output Markdown prose (no JSON).

CITATIONS:
- You may cite ONLY the evidence ids listed under ALLOWED EVIDENCE IDS, copied
  character for character. Never shorten, rename or invent an id.
- Put the citation immediately after the sentence it supports: [[ev:<evidence_id>]].
- If a sentence answers a checklist item, also tag it: [[req:<checklist_id>]].

FACTS:
- A factual statement about our firm must come from ONE cited passage.
- Copy every number, percentage, name, client and duration exactly as it appears
  in the passage you cite. Do not round, combine or restate figures.
- If the tender asks for a threshold (for example "greater than 30 percent") and
  no passage reaches it, do NOT state a figure that meets it. Write
  [EVIDENCE GAP: <the threshold and what the evidence actually shows>].

OTHER CONTENT:
- Proposed FUTURE actions use prospective phrasing ("We propose...", "In weeks 1-3
  the team will...") and carry NO evidence citation.
- If evidence is insufficient for a checklist item, write exactly:
  [EVIDENCE GAP: <what is missing>]
- For pricing / named staffing / commercial terms write exactly:
  **HUMAN INPUT REQUIRED:** <what the engagement partner must supply>
Keep it tight: 2-5 sentences per checklist item."""


def draft_prompt_version() -> str:
    return (config.DRAFT_PROMPT_VERSION or "v1").lower()


def draft_section(section_title, rfp_data, section_checklist, evidence_by_checklist,
                  *, all_supported_evidence_ids=None, section_evidence_pool=None,
                  complete=None) -> str:
    pool = section_evidence_pool or []
    ev_lines = []
    for e in pool:
        ev_lines.append(f"- evidence_id={e['evidence_id']} source={e['source_id']} "
                        f"category={e.get('category')}\n  \"{e['chunk_text'].strip()}\"")
    chk_lines = [f"- checklist_id={c['checklist_id']} handling={c['handling']}: "
                 f"{c['requirement_text']}" for c in section_checklist]
    user = (
        f"SECTION: {section_title}\n\n"
        f"CLIENT: {rfp_data.get('client')}\nTIMELINE: {rfp_data.get('timeline')}\n"
        f"PROBLEM: {rfp_data.get('problem_statement')}\n\n"
        f"CHECKLIST ITEMS FOR THIS SECTION:\n" + ("\n".join(chk_lines) or "(none)") +
        f"\n\nSUPPLIED EVIDENCE (verbatim -- cite by evidence_id):\n" +
        ("\n".join(ev_lines) or "(none)")
    )
    if section_title == "Executive Summary":
        user += "\n\nWrite this LAST: synthesise the rest; no new claims."
    system = _DRAFT_SYS
    if draft_prompt_version() == "v2":
        system = _DRAFT_SYS_V2
        allowed = sorted({e["evidence_id"] for e in pool})
        user += ("\n\nALLOWED EVIDENCE IDS (cite only these, exactly):\n"
                 + ("\n".join(f"- {i}" for i in allowed) or "(none: cite nothing)"))
    return complete(system, user).strip()


# --------------------------------------------------------------------------- #
# 4. Claim decomposition
# --------------------------------------------------------------------------- #
_SPLIT_SYS = """Split the given proposal section into atomic, independently-checkable
statements. Return ONLY a JSON list of strings, one atomic claim each. Split
compound sentences ("X and reduced Y by 5%") into separate items. Preserve any
[[ev:...]] and [[req:...]] tags on the item they belong to. Do not paraphrase
numbers or names."""


# A present-tense assertion about the firm is a factual claim even with no
# past-tense achievement verb in it. "VCG is ISO 27001 certified" and "VCG
# operates in 40 countries" previously classified as not-requiring-verification,
# so an unsupported credential could reach an approved proposal unchecked --
# exactly the failure the product exists to prevent.
# Sentences that talk about the proposal rather than asserting anything about
# the firm. These are the genuine exemptions; everything else a firm says about
# itself is a claim.
_CREDENTIAL_RE = re.compile(
    r"\b(certifie[ds]|certification|accredit\w*|attestation|iso\s*\d+|soc\s*2|"
    r"licen[cs]ed?|registered|member of|ranked|awarded|award|partner of|"
    r"operates?|employs?|maintains?|holds?)\b",
    re.I,
)


def decompose_claims(section_title: str, section_markdown: str, *, complete=None) -> list[dict]:
    try:
        raw = complete(_SPLIT_SYS, section_markdown)
        atoms = _json(raw)
        if not isinstance(atoms, list):
            raise ValueError
    except Exception:
        # fall back to deterministic sentence split -- never fail the pipeline
        atoms = sentences(section_markdown)

    out: list[dict] = []
    n = 0
    for atom in atoms:
        if not isinstance(atom, str):
            continue
        if atom.lstrip().startswith("[EVIDENCE GAP") or "HUMAN INPUT REQUIRED" in atom:
            continue
        cites = mock_llm._CITE_RE.findall(atom)
        req_ids = mock_llm._REQ_RE.findall(atom)
        clean = mock_llm._TAG_RE.sub("", atom).strip()
        for frag in split_compound(clean):
            frag = frag.strip()
            if len(frag) < 8:
                continue
            n += 1
            numeric = [t.raw for t in extract_numeric_tokens(frag)]
            entities = extract_named_entities(frag)
            low = frag.lower()
            prospective = mock_llm.is_prospective(frag)
            out.append({
                "claim_id": f"CLM-{section_title[:3].upper()}-{n:03d}",
                "section_name": section_title,
                "claim_text": frag,
                "claim_type": mock_llm._claim_type(frag, prospective, numeric, entities),
                "source_checklist_id": req_ids[0] if req_ids else None,
                "cited_evidence_ids": list(dict.fromkeys(cites)),
                "numeric_tokens": numeric,
                "named_entities": entities,
                "context_qualifiers": extract_context_qualifiers(frag),
                "requires_verification": mock_llm.is_verifiable_assertion(
                    frag, numeric, entities),
            })
    return out
