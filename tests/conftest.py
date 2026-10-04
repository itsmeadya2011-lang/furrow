"""Shared fixtures: environment isolation, settings factories and a fake LLM client.

Three rules hold for the whole suite:

* No provider is ever contacted. Every LLM interaction goes through :class:`FakeLLM`,
  which has the same surface as :class:`furrow.llm.LLMClient` and no SDK behind it.
* ``furrow.config.settings`` is never mutated. Each test builds an explicit
  :class:`~furrow.config.Settings`, and an autouse guard fails the test if the
  singleton changed underneath it.
* The environment cannot leak in. ``FURROW_*`` and API-key variables are removed,
  ``HOME`` is redirected and the CWD is a temp directory, so a developer's ``.env``
  cannot change an outcome.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import socket
import sys
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from furrow import config
from furrow.config import Settings
from furrow.llm import LLMClient

SettingsFactory = Callable[..., Settings]
Handler = Callable[[str, "FakeLLM"], "str | Awaitable[str]"]

API_KEY_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")

#: Body of a stand-in test suite that reports one green test and exits 0.
PASSING_SUITE = "print('1 passed')"

#: Body of a stand-in test suite that reports failures and exits non-zero.
FAILING_SUITE = "print('1 failed, 2 passed')\nraise SystemExit(1)"


@dataclass(frozen=True)
class LLMCall:
    """One recorded call to :meth:`FakeLLM.complete`."""

    prompt: str
    system: str
    model: str | None


class FakeLLM:
    """Stand-in for :class:`furrow.llm.LLMClient` with no provider behind it.

    Exposes exactly what the agents use -- a real ``.settings`` attribute, an
    ``async complete(prompt, system="", model=None)`` and ``aclose()`` -- and records
    every call so a test can assert on prompt content, the model that was requested
    and how many attempts a retry loop made.
    """

    def __init__(
        self,
        settings: Settings,
        responses: list[str] | None = None,
        handler: Handler | None = None,
    ) -> None:
        self.settings = settings
        self.calls: list[LLMCall] = []
        self.closed = False
        self._responses = list(responses or [])
        self._handler = handler

    @property
    def prompts(self) -> list[str]:
        """Every prompt this client was asked, in order."""
        return [call.prompt for call in self.calls]

    @property
    def models(self) -> list[str | None]:
        """The ``model=`` argument of every call, in order."""
        return [call.model for call in self.calls]

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.calls.append(LLMCall(prompt=prompt, system=system, model=model))
        if self._handler is not None:
            outcome = self._handler(prompt, self)
            if isinstance(outcome, str):
                return outcome
            return await outcome
        if not self._responses:
            raise AssertionError("FakeLLM.complete was called but no response was scripted")
        return self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]

    async def aclose(self) -> None:
        self.closed = True

    def as_client(self) -> LLMClient:
        """This fake, typed as the client the agents and orchestrator accept."""
        return cast(LLMClient, self)


@pytest.fixture(autouse=True)
def scrubbed_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Hide ``FURROW_*``, API keys and the CWD from every test.

    ``Settings`` reads the environment and a ``.env`` in the CWD, so without this a
    developer's exported ``FURROW_*`` variable or checked-out ``.env`` could decide
    what the tests see.
    """
    for name in list(os.environ):
        if name.startswith("FURROW_") or name in API_KEY_VARS:
            monkeypatch.delenv(name, raising=False)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def no_outbound_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn an accidental outbound TCP connection into a test failure.

    The suite must be offline, and this is the evidence: nothing in it can reach a
    provider even by accident. Local sockets (``socketpair`` and friends, which asyncio
    uses for its self-pipe) are untouched.
    """
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def refuse(sock: socket.socket, address: Any, original: Callable[..., Any]) -> Any:
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError(f"a test tried to open a network connection to {address!r}")
        return original(sock, address)

    monkeypatch.setattr(
        socket.socket, "connect", lambda self, address: refuse(self, address, real_connect)
    )
    monkeypatch.setattr(
        socket.socket, "connect_ex", lambda self, address: refuse(self, address, real_connect_ex)
    )


@pytest.fixture(autouse=True)
def settings_singleton_untouched() -> Iterator[None]:
    """Fail any test that mutates the process-wide settings singleton."""
    before = config.settings.model_dump()
    yield
    after = config.settings.model_dump()
    assert after == before, f"the settings singleton was mutated: {before} -> {after}"


@pytest.fixture
def make_settings(tmp_path: Path) -> SettingsFactory:
    """Build an explicit :class:`Settings`, pointed at a temp workspace by default."""

    def _make(**overrides: Any) -> Settings:
        values: dict[str, Any] = {"workspace": tmp_path, "_env_file": None}
        values.update(overrides)
        return Settings(**values)

    return _make


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """An empty directory a test may treat as ``settings.workspace``."""
    return tmp_path


@pytest.fixture
def outside_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A sibling of :func:`workspace`, so "outside the workspace" can be tested."""
    return tmp_path_factory.mktemp("outside")


@pytest.fixture
def make_client(make_settings: SettingsFactory) -> Callable[..., FakeLLM]:
    """Build a :class:`FakeLLM`, with or without an explicit settings object."""

    def _make(settings: Settings | None = None, **kwargs: Any) -> FakeLLM:
        return FakeLLM(settings if settings is not None else make_settings(), **kwargs)

    return _make


@pytest.fixture
def suite_command(workspace: Path) -> Callable[..., str]:
    """Write a tiny Python file that imitates a test runner; return its command.

    The command is a two-word shell-free argv (``[sys.executable, script]``), so it
    exercises the real subprocess path without depending on ``pytest``/``npm`` being on
    PATH and without ever pointing the tester at this repository.
    """

    def _build(body: str = PASSING_SUITE, name: str = "fake_suite.py") -> str:
        script = workspace / name
        script.write_text(f"import sys\n{body}\n", encoding="utf-8")
        return " ".join([shlex.quote(sys.executable), shlex.quote(str(script))])

    return _build


def run_coroutine(coro: Any) -> Any:
    """Run *coro* to completion.

    ``pytest-asyncio`` is not installed in the working interpreter, so the suite has no
    ``asyncio_mode``; coroutines are driven from synchronous tests instead.
    """
    return asyncio.run(coro)
