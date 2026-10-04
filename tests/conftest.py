"""Shared pytest fixtures for the furrow test suite.

Kept deliberately small and side-effect free so later test waves can reuse it:

* Every fixture here is **opt-in** (nothing is ``autouse``), so no test module
  inherits behaviour it did not ask for and no module can be broken by another
  module's fixture ordering.
* Nothing in this file touches the network.  Provider SDK / ``httpx`` stubs live
  in the test modules that need them, next to the behaviour they describe.
* ``Settings`` reads ``os.environ`` (prefix ``FURROW_``) *and* a ``.env`` file in
  the current working directory at construction time.  The fixtures below strip
  the first and disable the second, which removes the suite's implicit coupling
  to the ambient environment and to the CWD.

There is deliberately no ``tests/__init__.py`` requirement here; the suite relies
on pytest's rootdir-based collection.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import pytest

from furrow.config import Settings
from furrow.llm import LLMClient

__all__ = ["isolated_env", "make_client", "make_settings"]


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Delete every ambient ``FURROW_*`` variable for the duration of one test.

    Returns the (already patched) ``monkeypatch`` so a test can additionally stub
    other variables, e.g. ``ANTHROPIC_API_KEY``.  The original values are restored
    by monkeypatch teardown, so no test depends on another's side effects.
    """
    for key in [name for name in os.environ if name.startswith("FURROW_")]:
        monkeypatch.delenv(key)
    return monkeypatch


@pytest.fixture
def make_settings(isolated_env: pytest.MonkeyPatch) -> Callable[..., Settings]:
    """Return a factory building ``Settings`` isolated from ambient env and ``.env``.

    ``_env_file=None`` is what stops a ``.env`` file sitting in the current working
    directory from leaking into the value under test; ``isolated_env`` handles the
    ``os.environ`` half.  Keyword overrides are passed straight through, so a test
    can request exactly the configuration it wants::

        settings = make_settings(provider=Provider.OPENAI, max_tokens=10)
    """

    def _make_settings(**overrides: Any) -> Settings:
        # ``_env_file`` is a ``BaseSettings`` constructor keyword that works at runtime
        # but that mypy cannot see on the generated ``__init__`` signature.
        return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]

    return _make_settings


@pytest.fixture
def make_client(make_settings: Callable[..., Settings]) -> Callable[..., LLMClient]:
    """Return a factory building an ``LLMClient`` bound to isolated settings.

    The client is never allowed to build a real SDK client here; tests that need
    one assign their own fake to ``client._anthropic`` / ``client._openai`` (both
    are read through the sync ``anthropic`` / ``openai`` properties).
    """

    def _make_client(**overrides: Any) -> LLMClient:
        return LLMClient(settings=make_settings(**overrides))

    return _make_client