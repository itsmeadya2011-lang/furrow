from __future__ import annotations

import os
from pathlib import Path

import pytest

from furrow.config import reset_settings_cache

# Bare provider keys that ``furrow.llm`` and ``furrow.cli.main`` fall back to
# when the matching ``FURROW_*`` setting is not configured.
_PROVIDER_KEY_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")


@pytest.fixture(autouse=True)
def _fresh_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> object:
    """Isolate every test from the ambient environment and the settings cache.

    ``get_settings()`` is an ``lru_cache``-backed singleton, so without this a
    test that sets ``FURROW_*`` env vars would leak into every test after it.

    The working directory is moved to an empty ``tmp_path`` *before* the cache is
    cleared, because ``Settings`` loads ``.env`` relative to the current
    directory: a stray ``.env`` in the checkout must not be able to satisfy the
    tests that assert an unset value. The bare provider keys are cleared too,
    since ``furrow.llm`` and ``furrow.cli.main`` read them straight from the
    environment when no ``FURROW_*`` key is set.
    """
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("FURROW_"):
            monkeypatch.delenv(key, raising=False)
    for name in _PROVIDER_KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()
