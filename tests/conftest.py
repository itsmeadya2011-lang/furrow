"""Shared fixtures and offline stubs for the Furrow test suite.

Two rules make this suite deterministic:

* **No network, ever.** :class:`FakeLLM` replaces :class:`furrow.llm.LLMClient`
  wholesale; its ``complete``/``agent_loop`` return scripted strings and count
  every call. The sandboxed file helpers are *delegated* to a real
  :class:`~furrow.llm.LLMClient` (which never talks to a provider) so path
  confinement under test is the production behaviour, not a re-implementation.
* **No process-wide leakage.** The CLI and the agents mutate the
  ``furrow.config.settings`` singleton in place, so an autouse fixture snapshots
  and restores it, and another strips ambient ``FURROW_*``/provider credentials
  from the environment.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

import pytest

from furrow import config
from furrow.config import Settings
from furrow.llm import LLMClient

#: Environment variables that could otherwise leak into a test.
_LEAKY_ENV_PREFIXES = ("FURROW_",)
_LEAKY_ENV_NAMES = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_BASE_URL",
    "OPENAI_BASE_URL",
    "OLLAMA_BASE_URL",
)


def make_settings(**overrides: Any) -> Settings:
    """Build a ``Settings`` that ignores ``.env`` and is safe to mutate."""
    overrides.setdefault("anthropic_api_key", "test-key")
    overrides.setdefault("openai_api_key", "test-key")
    return Settings(_env_file=None, **overrides)


class ToolScript:
    """One scripted worker turn: run these tool calls, then return ``summary``."""

    def __init__(self, *calls: tuple[str, dict[str, Any]], summary: str = "") -> None:
        self.calls = list(calls)
        self.summary = summary

    async def __call__(self, execute: Any) -> str:
        for name, tool_input in self.calls:
            await execute(name, tool_input)
        return self.summary


class FakeLLM:
    """Scripted stand-in for :class:`furrow.llm.LLMClient`.

    ``responses`` feed :meth:`complete` (popped in order); ``scripts`` feed
    :meth:`agent_loop` and may be plain strings or awaitables such as
    :class:`ToolScript` that drive the ``execute`` callback. ``fail_with`` makes
    every LLM entry point raise, which is how the agents' "never raise" contracts
    are exercised.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        responses: list[str] | None = None,
        scripts: list[Any] | None = None,
        fail_with: BaseException | None = None,
    ) -> None:
        self.settings = settings if settings is not None else config.settings
        self.responses = list(responses or [])
        self.scripts = list(scripts or [])
        self.fail_with = fail_with
        self.complete_calls: list[dict[str, Any]] = []
        self.agent_loop_calls: list[dict[str, Any]] = []
        self.reads: list[str] = []
        self.writes: list[tuple[str, str]] = []
        self._sandbox = LLMClient(self.settings)

    # -- scripting ------------------------------------------------------

    def queue(self, *responses: str) -> FakeLLM:
        self.responses.extend(responses)
        return self

    def script(self, *turns: Any) -> FakeLLM:
        self.scripts.extend(turns)
        return self

    @property
    def complete_call_count(self) -> int:
        return len(self.complete_calls)

    @property
    def agent_loop_call_count(self) -> int:
        return len(self.agent_loop_calls)

    # -- LLMClient surface ---------------------------------------------

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.complete_calls.append({"prompt": prompt, "system": system, "model": model})
        if self.fail_with is not None:
            raise self.fail_with
        if not self.responses:
            raise AssertionError(
                f"FakeLLM.complete() call #{len(self.complete_calls)} has no scripted response"
            )
        return self.responses.pop(0)

    async def agent_loop(
        self,
        prompt: str,
        tools: list[dict],
        execute: Any,
        system: str = "",
        model: str | None = None,
        max_turns: int | None = None,
    ) -> str:
        self.agent_loop_calls.append(
            {
                "prompt": prompt,
                "tools": tools,
                "system": system,
                "model": model,
                "max_turns": max_turns,
                "execute": execute,
            }
        )
        if self.fail_with is not None:
            raise self.fail_with
        if not self.scripts:
            raise AssertionError("FakeLLM.agent_loop() called with no scripted turn")
        turn = self.scripts.pop(0)
        if callable(turn):
            return str(await turn(execute))
        return str(turn)

    # -- sandboxed file helpers (delegated to the real client) ----------

    def resolve_path(self, path: str | Path) -> Path:
        return self._sandbox.resolve_path(path)

    async def read_file(self, path: str | Path) -> str:
        self.reads.append(str(path))
        return await self._sandbox.read_file(path)

    async def write_file(self, path: str | Path, content: str) -> None:
        self.writes.append((str(path), content))
        await self._sandbox.write_file(path, content)

    def list_files(self, directory: str | Path = ".") -> list[str]:
        return self._sandbox.list_files(directory)


class MinimalClient:
    """A client that only offers ``settings`` and ``agent_loop``.

    Used to drive :class:`~furrow.agents.worker.WorkerAgent`'s internal fallbacks:
    the worker must still confine every path to the workspace when the injected
    client provides no file helpers at all.
    """

    def __init__(
        self, settings: Settings | None = None, *, scripts: list[Any] | None = None
    ) -> None:
        self.settings = settings if settings is not None else config.settings
        self.scripts = list(scripts or [])

    async def agent_loop(
        self,
        prompt: str,
        tools: list[dict],
        execute: Any,
        system: str = "",
        model: str | None = None,
        max_turns: int | None = None,
    ) -> str:
        if not self.scripts:
            raise AssertionError("MinimalClient.agent_loop() called with no scripted turn")
        turn = self.scripts.pop(0)
        if callable(turn):
            return str(await turn(execute))
        return str(turn)


@pytest.fixture(autouse=True)
def hermetic_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip ambient configuration so tests see the documented defaults."""
    for key in list(os.environ):
        if key.upper().startswith(_LEAKY_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    for name in _LEAKY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if any test tries to open an outbound socket."""

    def deny(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("network access is forbidden in the Furrow test suite")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


@pytest.fixture(autouse=True)
def restore_global_settings() -> Any:
    """Snapshot/restore ``furrow.config.settings``; the CLI mutates it in place."""
    original = config.settings
    snapshot = {name: getattr(original, name) for name in Settings.model_fields}
    try:
        yield original
    finally:
        for name, value in snapshot.items():
            setattr(original, name, value)
        config.settings = original


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Settings:
    """A ``Settings`` whose workspace is an empty temporary directory."""
    (tmp_path / "src").mkdir()
    (tmp_path / "README.md").write_text("# scratch\n", encoding="utf-8")
    return make_settings(workspace=tmp_path)
