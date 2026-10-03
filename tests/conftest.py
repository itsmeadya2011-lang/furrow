"""Shared fixtures for the Furrow test suite.

Two hazards are neutralised here:

1. The shared virtualenv contains an editable install whose ``furrow`` package can
   resolve to a *different* agent's worktree.  Every test must exercise the
   checkout that owns this file, so the repository root is forced onto
   ``sys.path[0]`` and the resolved package is asserted at import time.  A
   mis-resolved install fails loudly instead of silently testing foreign code.
2. Real credentials and developer ``.env`` overrides must never leak into a test,
   so every ``FURROW_*`` / provider-key variable is removed before each test and
   ``Settings`` objects are built with ``_env_file=None``.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from furrow import config as _config
from furrow.llm import LLMClient

_RESOLVED_CONFIG = Path(_config.__file__).resolve()
if not str(_RESOLVED_CONFIG).startswith(str(REPO_ROOT) + os.sep):
    raise RuntimeError(
        f"furrow resolved to {_RESOLVED_CONFIG}, which is outside this checkout "
        f"({REPO_ROOT}). The editable install in the shared venv points at another "
        "worktree; tests would silently validate foreign code."
    )
assert isinstance(_config.settings, _config.Settings)  # the module singleton must import cleanly

# Environment variables that must never influence a test outcome.
_CREDENTIAL_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL")

# Substrings that identify which agent asked for a completion, for FakeLLM routing.
PLANNER_MARKER = "planning agent"
WORKER_MARKER = "worker agent"
TESTER_MARKER = "tester agent"

Response = str | BaseException | Callable[[str], str]


def make_settings(**overrides: Any) -> _config.Settings:
    """Build a ``Settings`` instance that ignores the environment and any ``.env``.

    ``_env_file=None`` disables the ``.env`` lookup configured on the model, so
    these objects only reflect the keyword arguments passed here.
    """
    return _config.Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


class FakeLLM(LLMClient):
    """A scripted :class:`LLMClient` that never touches the network.

    Responses are chosen in this order:

    * ``routes``: first entry whose key is a substring of the prompt wins.  This
      is how one fake serves the planner, the worker and the tester with
      different payloads -- ``{"planning agent": plan_json, ...}``.
    * ``responses``: a queue consumed one item per call, in order.
    * ``default``: returned once the queue is exhausted.
    """

    def __init__(
        self,
        responses: Iterable[Response] | None = None,
        *,
        routes: dict[str, Response] | None = None,
        default: str = "",
        settings: _config.Settings | None = None,
        delay: float = 0.0,
        track_concurrency: bool = False,
    ) -> None:
        super().__init__(settings=settings if settings is not None else make_settings())
        self.responses: list[Response] = list(responses or [])
        self.routes: dict[str, Response] = dict(routes or {})
        self.default = default
        self.delay = delay
        self.track_concurrency = track_concurrency
        # Every ``(prompt, model)`` pair the agents asked for.
        self.calls: list[tuple[str, str | None]] = []
        self.active = 0
        self.max_active = 0

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.calls.append((prompt, model))
        if self.track_concurrency:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            response = self._next_response(prompt)
            if isinstance(response, BaseException):
                raise response
            if callable(response):
                response = response(prompt)
            if self.delay:
                await asyncio.sleep(self.delay)
            return str(response)
        finally:
            if self.track_concurrency:
                self.active -= 1

    def _next_response(self, prompt: str) -> Response:
        for needle, response in self.routes.items():
            if needle in prompt:
                return response
        if self.responses:
            return self.responses.pop(0)
        return self.default

    @property
    def prompts(self) -> list[str]:
        return [prompt for prompt, _ in self.calls]

    @property
    def models(self) -> list[str | None]:
        return [model for _, model in self.calls]

    def calls_matching(self, needle: str) -> list[tuple[str, str | None]]:
        return [call for call in self.calls if needle in call[0]]


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip credentials and developer configuration from the environment."""
    for name in list(os.environ):
        if name.startswith("FURROW_"):
            monkeypatch.delenv(name, raising=False)
    for name in _CREDENTIAL_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FURROW_PROVIDER", "anthropic")


@pytest.fixture
def settings_factory():
    return make_settings


@pytest.fixture
def fake_client() -> FakeLLM:
    return FakeLLM(settings=make_settings())


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """A minimal project directory used as ``settings.workspace``."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# fixture project\n", encoding="utf-8")
    return tmp_path
