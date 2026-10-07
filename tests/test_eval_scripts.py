"""The prompt A/B harness and the real-error capture, driven by a scripted model.

No network: LLM.complete is replaced by a fake that behaves like a careless
drafter (one good citation, one invented id, one figure the passage lacks), so
the scripts' measurements can be checked against known answers.
"""
import json
import re
import sys
from pathlib import Path

import config
from services import llm as llm_mod

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import capture_real_errors  # noqa: E402
import prompt_ab  # noqa: E402

SEEN: list[tuple[str, str]] = []


def fake_complete(self, system, user):
    SEEN.append((system, user))
    ids = re.findall(r"evidence_id=(\S+)", user)
    if not ids:
        return "[EVIDENCE GAP: no evidence supplied]"
    good = ids[0]
    return (f"VCG reduced turnaround time by 42 percent in a lending engagement. [[ev:{good}]]\n"
            f"VCG has delivered work for this kind of client. [[ev:CASE_FAKE_999]]\n"
            "We propose a phased engagement.")


def _patch(monkeypatch):
    SEEN.clear()
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "LLM_ALL_STAGES", False)
    monkeypatch.setattr(llm_mod.LLM, "complete", fake_complete)


def test_prompt_ab_measures_both_variants(monkeypatch, tmp_path):
    _patch(monkeypatch)
    out = tmp_path / "ab"
    rc = prompt_ab.main(["--fixtures", "abc_bank_lending_transformation.md",
                         "--out", str(out)])
    assert rc == 0
    data = json.loads(out.with_suffix(".json").read_text())
    for v in ("v1", "v2"):
        s = data["summary"][v]
        assert s["invented_citations"] >= 1          # CASE_FAKE_999 is caught
        assert s["blocked"] >= 1                     # the 42 percent claim is caught
    assert "| Invented citation ids |" in out.with_suffix(".md").read_text()
    # v2 really sends a different prompt, including the allowed-id list
    assert any("ALLOWED CITATIONS" in u for _, u in SEEN)
    assert any("ALLOWED CITATIONS" not in u for _, u in SEEN)


def test_prompt_ab_refuses_mock(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "mock")
    assert prompt_ab.main(["--fixtures", "abc_bank_lending_transformation.md"]) == 2


def test_capture_lists_real_blocked_claims(monkeypatch, tmp_path):
    _patch(monkeypatch)
    out = tmp_path / "errors.md"
    rc = capture_real_errors.main(["--fixture", "abc_bank_lending_transformation.md",
                                   "--out", str(out)])
    assert rc == 0
    text = out.read_text()
    assert "CASE_FAKE_999" in text
    assert "42 percent" in text
