"""The hosted provider: one Messages API call, usage recorded, no SDK needed."""
import io
import json

import config
from services import llm as llm_mod


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_anthropic_call_records_usage(monkeypatch):
    sent = {}

    def fake_urlopen(req, timeout=0):
        sent["url"] = req.full_url
        sent["headers"] = {k.lower(): v for k, v in req.header_items()}
        sent["body"] = json.loads(req.data)
        return _Resp(json.dumps({
            "content": [{"type": "text", "text": "Drafted [[ev:CV_001]]."}],
            "usage": {"input_tokens": 1200, "output_tokens": 300},
        }).encode())

    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(llm_mod, "urlopen", fake_urlopen)
    events: list[dict] = []
    token = llm_mod.set_usage_sink(events)
    try:
        out = llm_mod.LLM(provider="anthropic", model="claude-haiku-5-5").complete("sys", "user")
    finally:
        llm_mod.reset_usage_sink(token)

    assert out == "Drafted [[ev:CV_001]]."
    assert sent["url"].endswith("/v1/messages")
    assert sent["headers"]["x-api-key"] == "test-key"
    assert sent["body"]["temperature"] == 0 and sent["body"]["system"] == "sys"
    assert events and events[0]["provider"] == "anthropic"
    assert events[0]["input_tokens"] == 1200 and events[0]["output_tokens"] == 300
    assert events[0]["duration_seconds"] is not None


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    try:
        llm_mod.LLM(provider="anthropic").complete("s", "u")
    except RuntimeError as exc:
        assert "ANTHROPIC_API_KEY" in str(exc)
    else:
        raise AssertionError("expected a RuntimeError")
