"""A .env copied from .env.example must load clean values, not inline comments."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_env_example_copied_as_env_loads_clean_values(tmp_path):
    # Run in a subprocess so this test's environment and config import stay isolated.
    env_file = ROOT / ".env"
    if env_file.exists():
        return  # never clobber a developer's real .env
    env_file.write_text((ROOT / ".env.example").read_text())
    try:
        code = ("import config; print(config.EMBEDDINGS_BACKEND, config.VECTORSTORE_BACKEND, "
                "config.WEB_SEARCH_PROVIDER, config.LLM_PROVIDER)")
        clean = {k: v for k, v in os.environ.items()
                 if k not in {"EMBEDDINGS_BACKEND", "VECTORSTORE_BACKEND",
                              "WEB_SEARCH_PROVIDER", "LLM_PROVIDER"}}
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=clean,
                             capture_output=True, text=True, check=True).stdout.split()
        assert out == ["tfidf", "local", "mock", "mock"]
    finally:
        env_file.unlink()
