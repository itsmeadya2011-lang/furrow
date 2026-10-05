"""Hermetic test setup shared by the whole suite.

Every test in this project reads configuration from somewhere: the process
environment, a `.env` file in the current working directory, or a cached
`Settings` object. Without isolation, a developer's real `ANTHROPIC_API_KEY` or a
stray `.env` can change what a test asserts, and a test that fails only on
someone else's laptop is worse than no test at all.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from furrow.config import get_settings

# Prefixes that must never leak in from the developer's shell. `FURROW_` covers
# furrow's own settings; the bare vendor names are matched because a real
# ANTHROPIC_API_KEY / OPENAI_API_KEY in the environment binds to Settings.
ENV_PREFIXES = ("FURROW_", "ANTHROPIC_", "OPENAI_")


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give each test clean environment variables and an empty settings cache.

    Names are collected from `os.environ` before deleting because deletion is
    the only way to observe that a variable existed, and `monkeypatch` restores
    everything at teardown. The cache is cleared before *and* after the test
    because a cached Settings is process-global: leaking one would let a test
    that mutated the environment silently influence the next test's
    `get_settings()`.

    The `.env` file is not neutralised here. Tests that assert on defaults
    construct `Settings(_env_file=None)`, which is explicit at the call site and
    survives the fixture ordering that `chdir` would not; tests that want to
    exercise `.env` parsing write their own file into `tmp_path`.
    """
    for key in [key for key in os.environ if key.startswith(ENV_PREFIXES)]:
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
