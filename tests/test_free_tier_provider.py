"""Free-tier hosted providers (Groq, Gemini) through the OpenAI-compatible endpoint."""
import io
import json
from email.message import Message
from urllib.error import HTTPError

import config
from services import llm as llm_mod


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _ok(text="Drafted [[ev:CV_001]].", pin=900, pout=200):
    return _Resp(json.dumps({
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": pin, "completion_tokens": pout},
    }).encode())


def _call(provider="groq"):
    events: list[dict] = []
    token = llm_mod.set_usage_sink(events)
    try:
        out = llm_mod.LLM(provider=provider,
                          model=config.OPENAI_COMPAT[provider]["model"]).complete("sys", "user")
    finally:
        llm_mod.reset_usage_sink(token)
    return out, events


def test_groq_request_shape_and_usage(monkeypatch):
    sent = {}

    def fake(req, timeout=0):
        sent["url"] = req.full_url
        sent["auth"] = dict(req.header_items()).get("Authorization")
        sent["body"] = json.loads(req.data)
        return _ok()

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(llm_mod, "urlopen", fake)
    out, events = _call("groq")
    assert out == "Drafted [[ev:CV_001]]."
    assert sent["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert sent["auth"] == "Bearer gsk_test"
    assert sent["body"]["model"] == "openai/gpt-oss-20b"
    assert sent["body"]["temperature"] == 0
    assert sent["body"]["reasoning_effort"] == "low"
    assert events[0]["provider"] == "groq"
    assert events[0]["input_tokens"] == 900 and events[0]["output_tokens"] == 200


def test_rate_limit_is_waited_out_not_fatal(monkeypatch):
    calls = {"n": 0}
    slept = []

    def fake(req, timeout=0):
        calls["n"] += 1
        if calls["n"] == 1:
            hdrs = Message()
            hdrs["retry-after"] = "3"
            raise HTTPError(req.full_url, 429, "Too Many Requests", hdrs, io.BytesIO(b"{}"))
        return _ok()

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(llm_mod, "urlopen", fake)
    import time
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    out, events = _call("groq")
    assert out and calls["n"] == 2 and slept == [3.0]
    assert events[0]["rate_limit_wait_seconds"] == 3.0


def test_gemini_uses_its_own_endpoint_and_key(monkeypatch):
    sent = {}

    def fake(req, timeout=0):
        sent["url"] = req.full_url
        sent["body"] = json.loads(req.data)
        return _ok()

    monkeypatch.setenv("GEMINI_API_KEY", "AIza_test")
    monkeypatch.setattr(llm_mod, "urlopen", fake)
    _call("gemini")
    assert sent["url"].startswith("https://generativelanguage.googleapis.com/")
    assert "reasoning_effort" not in sent["body"]


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    try:
        _call("groq")
    except RuntimeError as exc:
        assert "GROQ_API_KEY" in str(exc)
    else:
        raise AssertionError("expected a RuntimeError")
