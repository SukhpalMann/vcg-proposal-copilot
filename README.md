# Proposal Copilot

Turns an inbound RFP into a source-grounded, review-ready proposal in which
**every factual statement is traceable to the evidence it was written from** —
and any statement the evidence does not support is caught before a reviewer
sees it.

- **Live URL:** with a free Groq key in Streamlit secrets it drafts with
  **gpt-oss-20b**, an open-weight model, at no cost; without a key it falls back
  to the deterministic generator and says so in the sidebar. See
  [Hosted AI](#hosted-ai-the-public-url).
- **Local AI:** Gemma 3 on Ollama. No key, no network, nothing leaves the
  machine. See [Running with the local model](#running-with-the-local-model).

Two deployments of the same pipeline, because the trade-off is real: hosted is
fast and cheap per proposal; local keeps a confidential tender on the machine.

> **For the AAs: run it with live AI in five commands (Python 3.11+)**
>
> ```bash
> pip install -r requirements.txt
> python scripts/seed_corpus.py
> pytest -q                                   # 201 tests, no key needed
> cp .env.example .env                        # then edit .env:
> #   LLM_PROVIDER=groq
> #   GROQ_API_KEY=<replace with a free key from https://console.groq.com/keys>
> streamlit run app.py
> ```
>
> **The only key is `GROQ_API_KEY`, and it goes in `.env`** (or in Streamlit
> Cloud's Secrets). None is committed. Without it the app still runs, but the
> sidebar says **"Simulation mode"** and no AI is used. With it, the sidebar
> says **"Live AI: gpt-oss-20b on Groq"**. Pick the ABC Bank tender, press
> **Assess bid fit**, record a BID decision, and allow about a minute: the free
> tier rate-limits, and the app waits and retries automatically.

> "VCG" is a fictional firm. The evidence corpus and the ten tenders are
> synthetic and labelled as such. Nothing here depicts a real client,
> engagement or person.

---

## What it actually does

Five stages, with two human gates that cannot be bypassed in code:

| | Stage | Gate |
|---|---|---|
| 1 | **Intake** — parse the tender into requirements, each tied to a quotation located in the source document | |
| 2 | **Qualify** — score evidence coverage, recommend bid / no-bid | **A practice lead must record a decision, a name and a reason before anything is drafted** |
| 3 | **Draft** — retrieve firm evidence, rank it, write each section from the passages that survived | |
| 4 | **Verify** — numeric, attribution and context checks, by rule | |
| 5 | **Release** — section approval + partner commercial sign-off | **No export until every section is approved and every unsupported claim is resolved or overridden with a recorded justification** |

## The demonstration

Running the ABC Bank tender (`python scripts/run_demo.py`) produces, among
others, these rows — reproduced from a real run:

| Requirement | Evidence | Drafted statement | Verdict |
|---|---|---|---|
| Named team members with relevant lending operations experience | `CV_001` | Ananya Mehta, Partner, has 18 years of experience in banking and lending operations | **Substantiated** |
| Demonstrated experience redesigning retail lending operations | `CASE_BANK_001` | In a retail lending engagement in India, VCG reduced pilot approval turnaround time by 18 percent and reduced manual handoffs by 30 percent | **Substantiated** |
| Evidence of **greater than 30 percent** turnaround-time improvement | *(none)* | In a comparable engagement, VCG delivered a **35 percent** turnaround-time improvement | **Unsubstantiated** |

The true 18 percent statement is substantiated as a claim, but the ">30
percent" requirement it maps to rolls up as **Partial**, with the shortfall
stated on the row: a true figure below the tender's bar does not meet the bar.

> **Honesty note.** In the deterministic generator the 35 percent overclaim is
> *planted* (see the header of `services/mock_llm.py`) so the tests always have
> an error to catch. For a pitch, use a real one:
> `python scripts/capture_real_errors.py` lists every statement a real model
> drafted that the verifier blocked. See
> [Evaluating the model](#evaluating-the-model).

The last row of the table is the point. The tender demands a threshold the evidence base
cannot meet, so the drafter does what a writer under pressure to answer every
evaluation criterion does: it asserts a figure that clears the bar, with nothing
behind it. That claim is **built from the tender's own metric and threshold**,
not from a fixed sentence — a tender asking for ">25 percent reduction in
days-sales-outstanding" produces a claim about days-sales-outstanding. It
carries no citation, so verification rejects it as an orphan claim and it blocks
release.

---

## Quick start

Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/seed_corpus.py      # build the evidence index from data/corpus/
pytest -q                          # 201 tests
streamlit run app.py
```

In the app: pick a tender, **Assess bid fit**, record a practice-lead decision,
then review the draft, the traceability matrix and the evidence; approve each
section and the commercial reference to unlock export.

Command line, no browser:

```bash
python scripts/run_demo.py                                # ABC Bank, prints the matrix
python scripts/run_demo.py xyz_insurer_actuarial_ai.md    # capability gap -> go/no-go
python scripts/run_demo.py pqr_bank_procurement_heavy.md  # procurement-governed tender
python scripts/run_demo.py def_capital_no_rubric.md       # tender with no evaluation rubric
```

### API keys

**None are committed, and the local mode needs none.** The hosted mode needs
one free key (no card): create it at <https://console.groq.com/keys>.

| Where you run it | Put the key here | Setting |
|---|---|---|
| Your machine | `.env` (copy `.env.example`) | `LLM_PROVIDER=groq` and `GROQ_API_KEY=<your key>` |
| Streamlit Community Cloud | App settings → Secrets (template: `.streamlit/secrets.toml.example`) | same two lines |

Alternatives use the same pattern: `LLM_PROVIDER=gemini` + `GEMINI_API_KEY`
(free tier, <https://aistudio.google.com/apikey>) or `LLM_PROVIDER=anthropic` +
`ANTHROPIC_API_KEY` (paid).

`.env` and `.streamlit/secrets.toml` are git-ignored.

---

## Hosted AI (the public URL)

```bash
LLM_PROVIDER=groq GROQ_API_KEY=<your free key> streamlit run app.py
```

Drafting goes to **gpt-oss-20b on Groq's free tier** through its
OpenAI-compatible endpoint (standard library only, temperature 0, tokens and
latency recorded per call). gpt-oss-20b is an open-weight model, so the product
runs on open models in both modes: Gemma 3 locally, gpt-oss hosted. Every other
stage, including verification, is unchanged and never sees the model.

The free tier limits tokens per minute, and a proposal makes about seven
drafting calls in a row. When the provider answers "429 Too Many Requests", the
client waits the period the server asks for and retries (up to 90 s per call);
the wait is recorded separately from model time.

On Streamlit Cloud, paste the lines from `.streamlit/secrets.toml.example` into
the app's Secrets and reboot; `app.py` copies them into the environment before
configuration loads. Uploaded documents are sent to the provider in this mode;
the sample tenders are synthetic.

---

## Running with the local model

```bash
ollama pull gemma3                 # one-time download, needs internet
LLM_PROVIDER=ollama LLM_MODEL=gemma3:latest streamlit run app.py
```

After the pull, inference talks only to `127.0.0.1:11434` — disconnect the
network and the product still works.

### Which stages use the model, and why

`config.provider_for()` routes per stage. This is a measured decision, not a
preference:

Whichever provider is active (`ollama` locally, `groq` hosted), routing is the
same:

| Stage | Handler | Why |
|---|---|---|
| Extract, plan | Deterministic parsing behind the source-span gate | Routing these through a 4B local model was **worse**: bid fit scored 0%, no claim was substantiated, and 70 citations resolved to nothing, because model-extracted requirements retrieved no evidence |
| **Draft sections** | **Local model** | Generation is where a model genuinely earns its place |
| Decompose claims | Deterministic splitter | Through the model this hung for 14 minutes on 1.4 seconds of CPU — a small model given a bare JSON-array schema has no natural stopping point. Splitting prose into sentences is mechanical |
| **Verify** | **Deterministic rules — never a model** | This is the product's whole thesis |

`LLM_ALL_STAGES=true` routes everything through the model and reproduces the
measurement above.

### Recording a run for a demonstration

A full local pass takes minutes — too slow to perform live. Record it once, then
reopen it instantly from the sidebar and perform one short live action on top:

```bash
LLM_PROVIDER=ollama LLM_MODEL=gemma3:latest python scripts/record_demo_run.py
```

A measured run on an M4 / 16 GB laptop: **7 calls, 10,474 input and 1,924 output
tokens, 250 s, 5 statements substantiated and 9 blocked.**

---

## How it works

```
intake -> extract + validate (source-span gate; ungrounded requirements dropped)
  -> plan -> retrieve -> rank (select / reject, each with a recorded reason)
  -> QUALIFY ------- practice-lead BID / NO-BID gate -------+
                                                            |
  -> detect conflicts -> optional external context (off by default)
  -> draft -> decompose into atomic claims -> VERIFY -> build traceability
  -> section review + partner commercial sign-off -> export
```

A plain ordered function chain; every stage mutates one `ProposalAgentState` and
appends to `execution_log`.

**The model interprets and writes. It never judges whether its own output is
true.** That is [`pipeline/verification.py`](pipeline/verification.py), which is
extractive and rule-based:

- **numeric** — every figure in a claim must have a rounding-tolerant match in
  the cited passage *whose plus/minus 20-token context is topically consistent*.
  This is what catches a "35%" that exists in the corpus only in an unrelated
  document about office electricity.
- **attribution** — valid by default; invalid only on a real contradiction. The
  facts are read from the firm's own CV documents, so adding a CV adds a
  checkable person.
- **context** — a claim's geography or industry qualifier must not be
  contradicted by the cited passage's metadata.
- A contradicted figure can **never** be recorded as substantiated, at any
  confidence.

Call it **evidence-consistency verification**, not fact-checking. It catches the
error categories the corpus is built to surface and nothing beyond that.

### Every drafted sentence is composed from a cited passage

Section text is not templated. Phase names come from a retrieved methodology
passage and the engagement length from the tender; each person's name, role and
tenure are parsed from their own CV passage; the risk register comes from a
past-engagement passage that enumerates delivery risks; figures are read out of
the passage being cited, so a claim and its citation cannot drift apart.

**Where no passage supports a section, the drafter writes an
`[EVIDENCE GAP: ...]` rather than prose about an industry the tender may have
nothing to do with.** Feeding it a hospital revenue-cycle tender yields evidence
gaps in Approach, Team, Risks and the Executive Summary — not a lending
methodology. Two regression tests enforce this.

---

## Cost

The Execution tab reports measured tokens and latency for the run, its rupee
cost, every hosted tier side by side, a 10,000-user monthly projection and a
"what we would change at that scale" list derived from those numbers. Tokens
and time are **measured**; every rate is a **declared assumption** shown beside
the figures and set in `config.py` (hosted rates checked against the Anthropic
and Groq pricing pages on 2026-10-08).

Using the measured local run above (10,474 input + 1,924 output tokens per
proposal; ₹89 per USD):

| | Per proposal | 10,000 proposals / month |
|---|---|---|
| gpt-oss-20b on Groq, **free tier** (what the demo runs) | ₹0 | ₹0 up to the free limits (about 15 proposals a day) |
| gpt-oss-20b on Groq, paid ($0.075 / $0.30 per M tokens) | ₹0.12 | ₹1,213 |
| Claude Haiku 5.5 ($0.10 / $0.50 per M tokens) | ₹0.18 | ₹1,788 |
| Claude Sonnet 5.5 ($2 / $10 per M tokens) | ₹3.58 | ₹35,767 |
| Local laptop (30 W × 250 s, ₹8/kWh) | ₹0.017 | does not scale on one laptop |
| Self-hosted GPU (₹110/h; 40 output tok/s single stream) | n/a | 148 busy GPU-hours (₹16,297), but **₹80,300** for one GPU always on |

What we would change at that scale:

- **Serve the hosted tier by default, on paid Groq once past the free limits.**
  One always-on GPU costs about 66 times the paid Groq bill; self-hosting only
  pays past roughly 660,000 proposals a month.
- **Keep local mode as the data-residency option**, priced as such: for a firm
  whose tenders may not leave its network, the GPU floor is the price of the
  guarantee.
- **Cache the shared prompt prefix.** 84% of tokens are input (tender plus
  evidence, resent per section); cached reads bill at a fraction of input.
- **Skip the model call for sections that will be an evidence gap.** The
  verifier would block that text anyway.
- **Keep verification rule-based.** It costs no tokens, so cost grows only with
  drafting.

The earlier projection multiplied the *laptop's* 250 seconds by 10,000 sessions,
which priced a rented GPU at laptop speed. GPU time is now derived from tokens.

---

## Evaluating the model

Two scripts make the AI claims checkable. Both refuse the deterministic
generator, because its errors are planted.

**Prompt A/B.** `DRAFT_PROMPT_VERSION` selects the drafting prompt. `v2` was
written against the failures observed with the local model: it lists the only
evidence ids the model may cite, requires figures to be copied verbatim from the
cited passage, and turns an unreachable tender threshold into an evidence-gap
marker instead of a rounded-up claim.

```bash
LLM_PROVIDER=groq GROQ_API_KEY=<free key> python scripts/prompt_ab.py
LLM_PROVIDER=ollama LLM_MODEL=gemma3:latest python scripts/prompt_ab.py   # slower
```

Both prompts draft the same three tenders; the rule-based verifier scores the
output (invented citation ids, uncited factual claims, blocked figures,
substantiation rate, tokens, cost). Results land in `docs/prompt_ab_results.md`.
Adopt v2 by setting `DRAFT_PROMPT_VERSION=v2` only if the table says it is
better.

**Real errors for the demo.** `python scripts/capture_real_errors.py
--run-id demo-local` (or `--fixture <tender>` to draft fresh) writes
`docs/real_model_errors.md`: each statement a real model drafted that the
verifier blocked, what it cited, and why it was refused.

---

## Known limitations

Stated plainly, because the product's whole claim is that it does not overstate:

- **Retrieval is weak.** TF-IDF barely separates good evidence from bad, so bid
  fit reads lower than it should. `EMBEDDINGS_BACKEND=sentence-transformers`
  improves it but needs a model download.
- **The local model invents citation IDs.** They are all rejected, which is the
  system working, but a run produces a substantial number of them.
- **Confidence is not calibrated.** It is a weighted blend of the signals, not a
  probability, despite being rendered as a bar.
- **No per-user isolation.** Saved runs are shared across browser sessions on a
  single deployment; the hosted URL is a single-tenant demonstration.
- **Chunking will mishandle tables.** Fixed-size splitting on headings; real
  tender eligibility criteria often live in tables.
- **No prompt-injection boundary.** Tender text reaches the drafting prompt
  untreated when a model is in use.
- **The corpus is synthetic**, so the demonstration is illustrative rather than a
  measurement against independently labelled ground truth.

---

## Backends

| Concern | Default | Alternative |
|---|---|---|
| Orchestrator | plain ordered function chain | — |
| Generation | `LLM_PROVIDER=mock` (deterministic) | `ollama` for local AI; `groq` (free, gpt-oss-20b) for the hosted demo; `gemini` (free); `anthropic` (paid); `litellm` |
| Embeddings | `tfidf` — fit on the seeded corpus, no download | `sentence-transformers` |
| Vector store | local numpy matrix, pickle-persisted | ChromaDB |
| External context | `mock`, disabled | `tavily` or `ddg` — background only, never cited as firm evidence |

`scripts/calibrate.py` scores known-good and known-bad pairs against the real
corpus and embedding backend and writes the verification thresholds into `.env`.

## Resumability

Every stage persists a JSON snapshot and a full pickled state to SQLite.
`pipeline.graph.load_run(run_id)` reopens a run fully working — review,
regeneration and export all continue. Review actions re-persist, so a reopened
run reflects approval progress. The sidebar has a **Reopen run** selector.

## Layout

```
config.py                  thresholds, model routing, cost assumptions, paths
models/schemas.py          Pydantic v2 models
state/graph_state.py       ProposalAgentState
data/corpus/*.md           10 synthetic evidence documents
data/sample_systems/*.json fictional CRM / HR / rate-card / time-billing inputs
fixtures/rfp/*.md          10 tenders (happy path, capability gap, procurement, no rubric,
                           off-domain, word-number and aggressive thresholds, ...)
pipeline/                  one module per stage, plus graph.py, qualification.py,
                           verification.py, review.py, export.py
services/                  corpus loader, embeddings, vector store, LLM providers,
                           costing, persistence, text utilities
scripts/                   seed_corpus, calibrate, run_demo, record_demo_run,
                           prompt_ab, capture_real_errors, stress_test
tests/                     201 tests
app.py, ui.py              Streamlit interface and design tokens
```

## Guarantees

- Every extracted requirement carries a locatable source quotation; ungrounded
  ones are dropped, not passed downstream. A document yielding no requirements
  stops the run rather than producing a proposal.
- Selected **and** rejected evidence both carry reasons; conflicts are surfaced,
  never silently resolved.
- Every citation must resolve to selected evidence — including on
  forward-looking statements, which need no evidence but may not assert false
  provenance.
- No code path drafts before a recorded practice-lead decision. No code path
  exports without every section approved and a partner commercial sign-off;
  unresolved gaps block release unless overridden with a recorded justification.
- The system never transmits or submits anything externally. Submission remains
  a manual action outside the platform.
