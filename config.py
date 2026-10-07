"""Central configuration. Everything overridable via environment / .env.

No model names, thresholds, or paths are hardcoded elsewhere in the codebase --
they are read from here so the System Owner (see SPEC.md Section 2) owns them.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


def _get(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _get_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _get_bool(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# --- Paths ------------------------------------------------------------------
CORPUS_DIR = ROOT / "data" / "corpus"
FIXTURE_DIR = ROOT / "fixtures" / "rfp"
DATA_DIR = ROOT / ".data"
VECTORSTORE_PATH = DATA_DIR / "vectorstore"
DB_PATH = Path(_get("DB_PATH", str(DATA_DIR / "proposal_copilot.sqlite")))
CHROMA_PATH = Path(_get("CHROMA_PATH", str(ROOT / ".chroma")))

# --- LLM ------------------------------------------------------------------
LLM_PROVIDER = _get("LLM_PROVIDER", "mock")   # mock | ollama | groq | gemini | anthropic | litellm
ANTHROPIC_API_KEY = _get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MAX_TOKENS = int(_get_float("ANTHROPIC_MAX_TOKENS", 1500))

# Hosted providers with a FREE tier, reached through their OpenAI-compatible
# chat endpoints (standard library only). Groq is the default for the public
# demo: no card needed, and it serves an open-weight model (gpt-oss-20b), so the
# product runs on open models both locally (Gemma) and hosted.
# Paid rates are listed so the cost slide can show what the same tokens would
# cost once past the free tier (Groq models page, checked 2026-10-08).
OPENAI_COMPAT = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": "openai/gpt-oss-20b",
        "label": "gpt-oss-20b on Groq",
        "usd_per_mtok": (0.075, 0.30),
        "extra": {"reasoning_effort": "low"},   # gpt-oss: fewer hidden tokens
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-2.5-flash",
        "label": "Gemini 2.5 Flash",
        "usd_per_mtok": None,                   # free tier only; not priced here
        "extra": {},
    },
}
FREE_TIER_PROVIDERS = {"groq", "gemini"}
COMPAT_MAX_TOKENS = int(_get_float("COMPAT_MAX_TOKENS", 2000))
COMPAT_MAX_RETRY_SECONDS = _get_float("COMPAT_MAX_RETRY_SECONDS", 90.0)


def compat_api_key(provider: str) -> str:
    preset = OPENAI_COMPAT.get(provider)
    return os.environ.get(preset["key_env"], "") if preset else ""


# A hosted deployment missing its key falls back to the deterministic generator
# and says so, rather than failing every run.
PROVIDER_FALLBACK_REASON = ""
if LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
    LLM_PROVIDER, PROVIDER_FALLBACK_REASON = "mock", "ANTHROPIC_API_KEY is not set"
elif LLM_PROVIDER in OPENAI_COMPAT and not compat_api_key(LLM_PROVIDER):
    PROVIDER_FALLBACK_REASON = f"{OPENAI_COMPAT[LLM_PROVIDER]['key_env']} is not set"
    LLM_PROVIDER = "mock"
_DEFAULT_MODEL = {"ollama": "gemma3:latest", "anthropic": "claude-haiku-5-5",
                  **{k: v["model"] for k, v in OPENAI_COMPAT.items()}}
LLM_MODEL = _get("LLM_MODEL", _DEFAULT_MODEL.get(LLM_PROVIDER, "anthropic/claude-haiku-5-5"))

# Drafting prompt version: v1 (original) or v2 (explicit allowed ids, verbatim
# figures, threshold -> gap). scripts/prompt_ab.py measures the difference.
DRAFT_PROMPT_VERSION = _get("DRAFT_PROMPT_VERSION", "v1")

OLLAMA_NUM_CTX = int(_get_float("OLLAMA_NUM_CTX", 8192))   # 4096 truncates drafting prompts

# Per-stage provider routing. Extraction and planning are structural work with a
# source-span validation gate behind them; a small local model made them WORSE,
# producing requirements that retrieved nothing. Generation is where a model
# genuinely earns its place, so drafting and claim decomposition follow
# LLM_PROVIDER while the structural stages stay deterministic unless explicitly
# overridden with LLM_ALL_STAGES=true.
LLM_ALL_STAGES = _get_bool("LLM_ALL_STAGES", False)
# Drafting only. Claim decomposition was measured running through the model and
# a small model given a bare JSON-array schema has no natural stopping point --
# it burned the full 180s timeout per section while the deterministic splitter
# does the same job instantly and reliably. Splitting prose into sentences is
# mechanical; writing the prose is not.
_GENERATIVE_STAGES = {"draft_sections"}


def provider_for(stage: str) -> str:
    """Which provider handles a given pipeline stage."""
    if LLM_PROVIDER == "mock" or LLM_ALL_STAGES:
        return LLM_PROVIDER
    return LLM_PROVIDER if stage in _GENERATIVE_STAGES else "mock"


# --- Embeddings --------------------------------------------------------------
EMBEDDINGS_BACKEND = _get("EMBEDDINGS_BACKEND", "tfidf")
EMBEDDINGS_MODEL = _get("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Vector store ---------------------------------------------------------
VECTORSTORE_BACKEND = _get("VECTORSTORE_BACKEND", "local")

# --- Web enrichment -----------------------------------------------------
WEB_SEARCH_PROVIDER = _get("WEB_SEARCH_PROVIDER", "mock")
WEB_SEARCH_ENABLED = _get_bool("WEB_SEARCH_ENABLED", False)

# --- Retrieval / ranking ------------------------------------------------
CHUNK_TARGET_CHARS = 700
CHUNK_OVERLAP_CHARS = 100
RETRIEVAL_TOP_K = 6
MAX_RETRIEVAL_PASSES = 2
MAX_VALIDATION_RETRIES = 2

# rerank weights (SPEC Section 11)
W_SEMANTIC = 0.65
W_LEXICAL = 0.20
W_METADATA = 0.15

# selection thresholds -- STARTING POINTS. Calibrated by scripts/calibrate.py
# against the real seeded corpus + embedding backend. Do not treat as settled.
SELECT_THRESHOLD = _get_float("SELECT_THRESHOLD", 0.30)
PARTIAL_THRESHOLD = _get_float("PARTIAL_THRESHOLD", 0.18)

# --- Deterministic verifier thresholds (Phase 0 calibration) ------------
# Defaults below are the values scripts/calibrate.py produced against the seeded
# corpus + the offline TF-IDF backend (see git history / run calibrate.py to
# re-derive). .env overrides them. With TF-IDF the claim<->chunk semantic band is
# low, so SUPPORTED/GAP leans on the numeric / attribution / context rules
# (C/D/E); the F/G similarity thresholds are the fallback.
SEM_SUPPORTED = _get_float("SEM_SUPPORTED", 0.116)
SEM_PARTIAL = _get_float("SEM_PARTIAL", 0.058)
LEX_SUPPORTED = _get_float("LEX_SUPPORTED", 0.218)

# numeric matcher tolerance (SPEC Section 15)
NUMERIC_ABS_TOLERANCE = 0.5          # percentage points / absolute units
NUMERIC_CONTEXT_WINDOW = 20          # +/- tokens around a number when checking consistency

# --- Cost model (services/costing.py) -------------------------------------
# ASSUMPTIONS, not measurements. costing.assumptions() surfaces every one of
# these next to the numbers. Rates checked on 2026-10-08 against
# https://console.groq.com/docs/models and
# https://platform.claude.com/docs/en/about-claude/pricing (prompts under 100k
# tokens). Re-check before quoting a figure.
USD_INR = _get_float("USD_INR", 89.0)                    # exchange rate
# Comparison tiers shown alongside, so the quality / cost trade-off is visible.
HOSTED_TIERS = {
    "gpt-oss-20b on Groq": (0.075, 0.30),
    "Claude Haiku 5.5": (0.10, 0.50),
    "Claude Sonnet 5.5": (2.00, 10.00),
}
# Primary hosted model: the one the public demo runs. Follows the provider.
_PRIMARY = {"groq": "gpt-oss-20b on Groq", "anthropic": "Claude Haiku 5.5"}
HOSTED_MODEL_LABEL = _get("HOSTED_MODEL_LABEL",
                          _PRIMARY.get(LLM_PROVIDER, "gpt-oss-20b on Groq"))
_rates = HOSTED_TIERS.get(HOSTED_MODEL_LABEL, HOSTED_TIERS["gpt-oss-20b on Groq"])
API_USD_PER_MTOK_INPUT = _get_float("API_USD_PER_MTOK_INPUT", _rates[0])
API_USD_PER_MTOK_OUTPUT = _get_float("API_USD_PER_MTOK_OUTPUT", _rates[1])
LOCAL_DEVICE_WATTS = _get_float("LOCAL_DEVICE_WATTS", 30.0)   # laptop draw while generating
ELECTRICITY_INR_PER_KWH = _get_float("ELECTRICITY_INR_PER_KWH", 8.0)
GPU_INR_PER_HOUR = _get_float("GPU_INR_PER_HOUR", 110.0)      # rented inference GPU
# Self-hosted throughput for a 4B model on one rented inference GPU, single
# stream, no batching (conservative). The old projection reused the LAPTOP's
# wall-clock seconds, which overstated GPU hours several-fold.
GPU_OUTPUT_TOK_PER_SEC = _get_float("GPU_OUTPUT_TOK_PER_SEC", 40.0)
GPU_PREFILL_TOK_PER_SEC = _get_float("GPU_PREFILL_TOK_PER_SEC", 2000.0)
GPU_HOURS_PER_MONTH = 730.0          # one GPU rented around the clock

# --- Export gating ------------------------------------------------------
ALLOW_GAP_OVERRIDE = True            # unresolved GAPs block export unless explicitly overridden w/ reason


def ensure_dirs() -> None:
    for p in (DATA_DIR, VECTORSTORE_PATH, DB_PATH.parent):
        p.mkdir(parents=True, exist_ok=True)
