"""LLM access.

Two providers:

* ``mock`` (default) -- deterministic, fixture-aware. Lets the whole pipeline,
  the demo, and the tests run fully offline. The mock is intentionally *not*
  smart: it does structural parsing and a couple of planted behaviours (notably
  the >30%% -> "35%%" overclaim) so the deterministic verifier has something real
  to catch.
* ``ollama`` -- a local model on 127.0.0.1; nothing leaves the machine.
* ``groq`` / ``gemini`` -- free-tier hosted models through their OpenAI-compatible
  endpoints, standard library only. Groq (gpt-oss-20b) is what the public demo
  uses; set GROQ_API_KEY in Streamlit secrets.
* ``anthropic`` -- Claude via the Anthropic Messages API, standard library only.
* ``litellm`` -- routes real calls through LiteLLM using LLM_MODEL from .env.
  Import-guarded; only used when explicitly configured.

Only these four operations are ever delegated to an LLM (SPEC Section 5 design
principle): interpret/extract, plan, draft, decompose-claims. The LLM is never
asked "is this claim true".
"""
from __future__ import annotations

import json
import re
from contextvars import ContextVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import config

_usage_sink: ContextVar[list[dict] | None] = ContextVar("model_usage_sink", default=None)
_current_stage: ContextVar[str] = ContextVar("model_stage", default="completion")


def set_usage_sink(events: list[dict]):
    return _usage_sink.set(events)


def reset_usage_sink(token) -> None:
    _usage_sink.reset(token)


def _record_usage(event: dict) -> None:
    sink = _usage_sink.get()
    if sink is not None:
        sink.append(event)


def _call_stage(complete, stage: str, system: str, user: str) -> str:
    token = _current_stage.set(stage)
    try:
        return complete(system, user)
    finally:
        _current_stage.reset(token)

# A bullet runs until the next bullet, a blank line, or a new heading -- NOT
# until the end of the line. Matching one line truncated every wrapped bullet:
# "Evidence of greater than 30 percent turnaround-time improvement in a"
# silently lost "comparable lending engagement." Real tenders wrap constantly,
# so this corrupted the requirement text, the retrieval query built from it, and
# the instruction handed to the drafter.
# Tenders number their criteria as often as they bullet them. Matching only
# "-" and "*" meant a numbered or lettered evaluation list extracted nothing at
# all, and the run proceeded with zero requirements from that section.
_MARKER = r"(?:[-*•]|\(?[0-9]{1,2}[.)]|\(?[a-z][.)])"
_BULLET = re.compile(
    rf"^[ \t]*{_MARKER}[ \t]+(.+?)(?=\n[ \t]*{_MARKER}[ \t]|\n[ \t]*\n|\n#|\Z)",
    re.M | re.S)
_H2 = re.compile(r"^##\s+(.*)$", re.M)
# A sentence boundary is a full stop, then whitespace, then a capital -- but a
# drafted sentence ends with its citation tag, so the tag sat between the stop
# and the capital and the boundary was never recognised. Three sentences became
# one 402-character "atomic" claim, and verifying any part of it stamped the
# whole blob SUPPORTED: a run had Ananya Mehta's CV substantiating "a key team
# member has previously led retail lending operations redesign for a large
# Indian bank", which that CV does not say. The tags belong to the sentence they
# follow, so the boundary is placed after them.
_TAG_RUN = r"(?:\s*\[\[[^\]]*\]\])*"
_SENT_SPLIT = re.compile(rf"(?<=[.!?])(?P<tags>{_TAG_RUN})\s+(?=[\"\'(\[]?[A-Z])")


# The model also makes the citation the grammatical subject:
#   "... across India. [[ev:CASE_BANK_001::00.00]] demonstrates our experience,
#    with Ananya Mehta ... [[ev:CV_001::01.00]] details a previous engagement
#    where ... [[ev:CASE_BANK_001::02.00]] highlights a pilot ..."
# Stripping the tag leaves a sentence starting with a lowercase verb, so the
# capital-letter requirement never fired and four assertions became one
# 650-character claim with one verdict. Here the tag belongs to the sentence that
# FOLLOWS it, so the boundary goes before the tags rather than after them.
_SENT_SPLIT_LOWER = re.compile(rf"(?<=[.!?])\s+(?={_TAG_RUN.strip()}\s*[a-z])")


def split_sentences(chunk: str) -> list[str]:
    """Split on sentence boundaries, keeping each sentence's citation tags.

    A trailing citation stays with the sentence it follows; a citation acting as
    the next sentence's subject goes with that sentence instead.
    """
    cuts = set()
    for m in _SENT_SPLIT.finditer(chunk):
        cuts.add((m.start() + len(m.group("tags")), m.end()))
    for m in _SENT_SPLIT_LOWER.finditer(chunk):
        cuts.add((m.start(), m.end()))

    out: list[str] = []
    last = 0
    for cut, resume in sorted(cuts):
        if cut < last:
            continue
        piece = chunk[last:cut].strip()
        if piece:
            out.append(piece)
        last = resume
    tail = chunk[last:].strip()
    if tail:
        out.append(tail)
    return out

PROSPECTIVE_MARKERS = (
    "we propose", "we will", "the team will", "in weeks", "during weeks",
    "week 1", "weeks 1", "our approach will", "we intend", "we would",
    "must be", "must submit", "is time-boxed", "will be delivered",
)


# --------------------------------------------------------------------------- #
# Public surface
# --------------------------------------------------------------------------- #
class LLM:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = provider or config.LLM_PROVIDER
        self.model = model or config.LLM_MODEL

    # -- delegated operations (SPEC Section 5) --------------------------
    # interpret/extract, plan, draft, decompose-claims. Never "is this true?".
    def extract_requirements(self, rfp_text: str, filename: str = "") -> dict:
        if self.provider == "mock":
            from services import mock_llm
            return mock_llm.extract_requirements(rfp_text, filename)
        from services import real_llm
        return real_llm.extract_requirements(
            rfp_text, filename,
            complete=lambda system, user: _call_stage(self.complete, "extract", system, user))

    def plan_response(self, rfp_data: dict) -> dict:
        if self.provider == "mock":
            from services import mock_llm
            return mock_llm.plan_response(rfp_data)
        from services import real_llm
        return real_llm.plan_response(
            rfp_data, complete=lambda system, user: _call_stage(self.complete, "plan", system, user))

    def draft_section(self, section_title: str, rfp_data: dict,
                      section_checklist: list[dict],
                      evidence_by_checklist: dict[str, list[dict]],
                      *, all_supported_evidence_ids: list[str] | None = None,
                      section_evidence_pool: list[dict] | None = None) -> str:
        if self.provider == "mock":
            from services import mock_llm
            return mock_llm.draft_section(
                section_title, rfp_data, section_checklist, evidence_by_checklist,
                all_supported_evidence_ids=all_supported_evidence_ids,
                section_evidence_pool=section_evidence_pool or [],
            )
        from services import real_llm
        return real_llm.draft_section(
            section_title, rfp_data, section_checklist, evidence_by_checklist,
            all_supported_evidence_ids=all_supported_evidence_ids,
            section_evidence_pool=section_evidence_pool or [],
            complete=lambda system, user: _call_stage(self.complete, "draft", system, user),
        )

    def decompose_claims(self, section_title: str, section_markdown: str) -> list[dict]:
        if self.provider == "mock":
            from services import mock_llm
            return mock_llm.decompose_claims(section_title, section_markdown)
        from services import real_llm
        return real_llm.decompose_claims(
            section_title, section_markdown,
            complete=lambda system, user: _call_stage(self.complete, "claim_split", system, user))

    # -- generic completion transport (real provider) ------------------
    def complete(self, system: str, user: str) -> str:
        stage = _current_stage.get()
        if self.provider == "ollama":
            from services.ollama_schemas import BY_STAGE
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                # num_ctx: the Ollama default of 4096 silently truncates a
                # drafting prompt that carries several evidence passages, which
                # makes the model invent rather than cite. temperature 0 keeps a
                # re-run reproducible.
                "options": {"temperature": 0, "num_ctx": config.OLLAMA_NUM_CTX},
            }
            if stage in BY_STAGE:
                payload["format"] = BY_STAGE[stage]
            request = Request(
                "http://127.0.0.1:11434/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=180) as response:
                    data = json.load(response)
            except (URLError, TimeoutError) as exc:
                raise RuntimeError(
                    "Local Ollama is unavailable. Start Ollama and confirm "
                    f"model {self.model!r} is installed: {exc}") from exc
            _record_usage({
                "stage": stage, "provider": "ollama", "model": self.model,
                "input_tokens": data.get("prompt_eval_count"),
                "output_tokens": data.get("eval_count"),
                "reasoning_tokens": None,
                "duration_seconds": round(data.get("total_duration", 0) / 1_000_000_000, 3),
            })
            return data.get("message", {}).get("content", "")
        if self.provider == "anthropic":
            return self._complete_anthropic(stage, system, user)
        if self.provider in config.OPENAI_COMPAT:
            return self._complete_openai_compat(stage, system, user)
        if self.provider != "litellm":
            raise RuntimeError(
                "LLM.complete() requires LLM_PROVIDER=ollama or litellm; the "
                "mock provider uses structured helpers instead."
            )
        try:
            import litellm
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "LLM_PROVIDER=litellm but the 'litellm' package is not installed "
                "(pip install litellm)."
            ) from exc
        resp = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = resp.get("usage") or {}
        _record_usage({
            "stage": stage, "provider": "litellm", "model": self.model,
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "duration_seconds": None,
        })
        return resp["choices"][0]["message"]["content"]

    def _complete_openai_compat(self, stage: str, system: str, user: str) -> str:
        """Chat completion on an OpenAI-compatible endpoint (Groq, Gemini).

        Free tiers enforce per-minute token limits, and one proposal makes about
        seven drafting calls in quick succession. A 429 is therefore expected,
        not exceptional: wait for the period the server asks for and retry,
        up to COMPAT_MAX_RETRY_SECONDS in total.
        """
        import time

        preset = config.OPENAI_COMPAT[self.provider]
        key = config.compat_api_key(self.provider)
        if not key:
            raise RuntimeError(f"LLM_PROVIDER={self.provider} but "
                               f"{preset['key_env']} is not set.")
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": config.COMPAT_MAX_TOKENS,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **preset.get("extra", {}),
        }
        body = json.dumps(payload).encode("utf-8")
        waited = 0.0
        started = time.monotonic()
        while True:
            request = Request(
                f"{preset['base_url']}/chat/completions", data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=120) as response:
                    data = json.load(response)
                break
            except HTTPError as exc:
                if exc.code != 429 or waited >= config.COMPAT_MAX_RETRY_SECONDS:
                    detail = ""
                    try:
                        detail = exc.read().decode("utf-8", "replace")[:300]
                    except Exception:
                        pass
                    raise RuntimeError(f"{preset['label']} call failed "
                                       f"(HTTP {exc.code}): {detail}") from exc
                try:
                    pause = float(exc.headers.get("retry-after") or 0)
                except (TypeError, ValueError):
                    pause = 0.0
                pause = min(max(pause, 2.0), 30.0)
                time.sleep(pause)
                waited += pause
            except (URLError, TimeoutError) as exc:
                raise RuntimeError(f"{preset['label']} call failed: {exc}") from exc
        usage = data.get("usage") or {}
        _record_usage({
            "stage": stage, "provider": self.provider, "model": self.model,
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "duration_seconds": round(time.monotonic() - started - waited, 3),
            "rate_limit_wait_seconds": round(waited, 1),
        })
        choice = (data.get("choices") or [{}])[0]
        return (choice.get("message") or {}).get("content") or ""

    def _complete_anthropic(self, stage: str, system: str, user: str) -> str:
        """One Messages API call, standard library only, with measured usage."""
        import time

        key = config.ANTHROPIC_API_KEY
        if not key:
            raise RuntimeError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set.")
        payload = {
            "model": self.model,
            "max_tokens": config.ANTHROPIC_MAX_TOKENS,
            "temperature": 0,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        request = Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        started = time.monotonic()
        try:
            with urlopen(request, timeout=120) as response:
                data = json.load(response)
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"Anthropic API call failed: {exc}") from exc
        usage = data.get("usage") or {}
        _record_usage({
            "stage": stage, "provider": "anthropic", "model": self.model,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "reasoning_tokens": None,
            "duration_seconds": round(time.monotonic() - started, 3),
        })
        return "".join(block.get("text", "") for block in data.get("content", [])
                       if block.get("type") == "text")


# --------------------------------------------------------------------------- #
# Structural parsing helpers (shared by mock + used to pre-parse for real LLM)
# --------------------------------------------------------------------------- #
def parse_sections(md: str) -> dict[str, str]:
    out: dict[str, str] = {}
    matches = list(_H2.finditer(md))
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        out[name.lower()] = md[start:end].strip()
    return out


def bullets(block: str) -> list[str]:
    """Bullets from a block, with wrapped continuation lines rejoined."""
    out = []
    for raw in _BULLET.findall(block):
        out.append(" ".join(raw.split()))      # collapse the wrap into one line
    return [b for b in out if b]


_LEADING_VERB = re.compile(
    r"^(?:to\s+)?(demonstrat\w*|provid\w*|redesign\w*|diagnos\w*|defin\w*|produc\w*|"
    r"quantif\w*|deliver\w*|reduc\w*|built|build|led|advis\w*|improv\w*|align\w*|"
    r"recommend\w*|assess\w*|prototyp\w*|design\w*|run|establish\w*|map)\b",
    re.I,
)


def split_compound(text: str) -> list[str]:
    """SPEC Section 8.2 / Section 14 -- split a compound requirement/claim.

    Deliberately conservative: only split a real enumeration -- clauses joined by
    ';' or by ', and '/', ' -- where each fragment begins with its own action
    verb or carries its own number. Never split a bare 'X and Y' noun pair such
    as 'credit policy and governance'.
    """
    parts = re.split(r"\s*;\s+|\s*,\s+and\s+|\s*,\s+(?=[a-z])", text.strip())
    parts = [p.strip(" .") for p in parts if p.strip(" .")]
    if len(parts) < 2:
        return [text.strip()]

    def standalone(frag: str) -> bool:
        f = frag.strip()
        if len(f) < 15:
            return False
        return bool(_LEADING_VERB.match(f)) or bool(re.search(r"\d", f))

    if all(standalone(p) for p in parts):
        return [p if p.endswith(".") else p + "." for p in parts]
    return [text.strip()]


def is_prospective(text: str) -> bool:
    low = text.lower()
    return any(mk in low for mk in PROSPECTIVE_MARKERS)


def sentences(text: str) -> list[str]:
    clean = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.M)  # drop headings
    out: list[str] = []
    for chunk in clean.split("\n"):
        chunk = chunk.strip().lstrip("-*").strip()
        if not chunk:
            continue
        out.extend(split_sentences(chunk))
    return out


def get_llm(stage: str | None = None) -> LLM:
    """Return the LLM for a stage, honouring config.provider_for()."""
    if stage is None:
        return LLM()
    return LLM(provider=config.provider_for(stage))
