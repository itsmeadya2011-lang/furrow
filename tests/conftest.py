"""Shared fixtures for the Furrow regression suite.

Four global safety nets are installed for every test:

* no process may open a non-loopback socket (proves the suite is hermetic);
* inherited ``FURROW_*`` environment variables are cleared, so ``Settings`` starts
  from a known state;
* the module level ``furrow.config.settings`` singleton must come out of the test
  unchanged;
* process wide logging state is restored afterwards.

On top of that, :func:`install_scripted` patches ``LLMClient.complete`` for the
tests that need deterministic agent replies.  The only tests that exercise the
real :meth:`LLMClient.complete` are in ``test_llm.py``, and they do it through
in-process fakes (``httpx.MockTransport`` and injected SDK stand-ins), never a
real provider.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import socket
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import structlog

from furrow.agents.prompts import (
    PLANNER_SYSTEM_PROMPT,
    TESTER_SYSTEM_PROMPT,
    WORKER_SYSTEM_PROMPT,
)
from furrow.agents.tester import TesterAgent
from furrow.config import Settings, TestResult
from furrow.config import settings as global_settings
from furrow.llm import LLMClient

# ``pytest`` tries to collect any ``Test*`` name imported into a test module.
# ``TestResult`` is a pydantic model and ``TesterAgent`` is the agent under test,
# not test cases: opt both out of collection.
TestResult.__test__ = False
TesterAgent.__test__ = False

PLANNER = "planner"
WORKER = "worker"
TESTER = "tester"

_BY_SYSTEM = {
    PLANNER_SYSTEM_PROMPT: PLANNER,
    WORKER_SYSTEM_PROMPT: WORKER,
    TESTER_SYSTEM_PROMPT: TESTER,
}

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "", "0.0.0.0", "::"})


class NetworkAccessDenied(RuntimeError):
    """Raised when a test tries to open a non-loopback connection."""


def _address_is_local(address: Any) -> bool:
    if not isinstance(address, tuple) or not address:
        return True
    host = address[0]
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    return str(host) in _LOOPBACK_HOSTS


# --------------------------------------------------------------------- networking


def _guard_family(family: int, address: Any) -> None:
    if family in (socket.AF_UNIX, socket.AF_UNSPEC):
        return
    if not _address_is_local(address):
        raise NetworkAccessDenied(f"test attempted real network I/O to {address!r}")


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that tries to talk to anything but loopback."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def connect(self: socket.socket, address: Any, *args: Any, **kwargs: Any) -> Any:
        _guard_family(self.family, address)
        return real_connect(self, address, *args, **kwargs)

    def connect_ex(self: socket.socket, address: Any, *args: Any, **kwargs: Any) -> Any:
        _guard_family(self.family, address)
        return real_connect_ex(self, address, *args, **kwargs)

    real_create_connection = socket.create_connection

    def create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        _guard_family(socket.AF_INET, address)
        return real_create_connection(address, *args, **kwargs)

    real_getaddrinfo = socket.getaddrinfo

    def getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
        if str(host) not in _LOOPBACK_HOSTS:
            raise NetworkAccessDenied(f"test attempted a DNS lookup for {host!r}")
        return real_getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "create_connection", create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


# ----------------------------------------------------------------- env isolation


@pytest.fixture(autouse=True)
def clean_furrow_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts without inherited ``FURROW_*`` configuration."""
    for name in [key for key in os.environ if key.startswith("FURROW_")]:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def isolated_singletons() -> Iterator[None]:
    """Fail loudly if a test mutates the module level ``settings`` singleton."""
    before = global_settings.model_dump()
    yield
    assert global_settings.model_dump() == before, "the global Settings singleton was mutated"


@pytest.fixture(autouse=True)
def restore_logging() -> Iterator[None]:
    """Undo the process wide logging changes ``configure_logging`` makes."""
    root = logging.getLogger()
    level = root.level
    handlers = list(root.handlers)
    processors = list(structlog.get_config()["processors"])
    yield
    root.setLevel(level)
    root.handlers = handlers
    structlog.configure(processors=processors)


# ------------------------------------------------------------------- LLM stubbing


@dataclass
class LLMCall:
    """One recorded call to the stubbed :meth:`LLMClient.complete`."""

    role: str
    system: str
    model: str | None
    prompt: str


def plan_reply(*tasks: dict[str, Any], rationale: str = "test plan") -> str:
    """A planner reply carrying *tasks* as a bare JSON object."""
    return json.dumps({"tasks": list(tasks), "rationale": rationale})


def worker_reply(summary: str = "did the work", files: list[dict[str, str]] | None = None) -> str:
    """A worker reply in the documented ``{summary, files}`` shape."""
    return json.dumps({"summary": summary, "files": list(files or [])})


def verdict_json(passed: bool, summary: str = "verdict", failures: list[str] | None = None) -> str:
    """A tester reply in the documented ``{passed, summary, failures}`` shape."""
    return json.dumps({"passed": passed, "summary": summary, "failures": list(failures or [])})


def task(task_id: str, description: str | None = None, **extra: Any) -> dict[str, Any]:
    """A planner-shaped task dict."""
    payload: dict[str, Any] = {"id": task_id, "description": description or f"task {task_id}"}
    payload.update(extra)
    return payload


class ScriptedCompletion:
    """Deterministic in-process replacement for :meth:`LLMClient.complete`.

    The reply is chosen from the agent system prompt, so the *real* planner,
    worker and tester agents - and the real pydantic models - run against it.
    A reply may be a string or a callable taking the recorded :class:`LLMCall`,
    which is how a test simulates a failing worker or a blocked run.
    """

    def __init__(
        self,
        planner: str | Callable[[LLMCall], Any] = "",
        worker: str | Callable[[LLMCall], Any] = "",
        tester: str | Callable[[LLMCall], Any] = "",
    ) -> None:
        self._replies = {PLANNER: planner, WORKER: worker, TESTER: tester}
        self.calls: list[LLMCall] = []
        self.by_role: dict[str, list[LLMCall]] = {PLANNER: [], WORKER: [], TESTER: []}

    async def __call__(self, prompt: str, system: str = "", model: str | None = None) -> str:
        role = _BY_SYSTEM.get(system, "other")
        call = LLMCall(role=role, system=system, model=model, prompt=prompt)
        self.calls.append(call)
        self.by_role.setdefault(role, []).append(call)
        reply = self._replies.get(role, "")
        if callable(reply):
            reply = reply(call)
        if inspect.isawaitable(reply):
            reply = await reply
        return reply

    # convenience accessors -------------------------------------------------
    def prompts(self, role: str) -> list[str]:
        return [call.prompt for call in self.by_role.get(role, [])]

    def models(self, role: str) -> list[str | None]:
        return [call.model for call in self.by_role.get(role, [])]

    def only_prompt(self, role: str) -> str:
        prompts = self.prompts(role)
        assert len(prompts) == 1, f"expected exactly one {role} call, got {len(prompts)}"
        return prompts[0]


def install_scripted(
    monkeypatch: pytest.MonkeyPatch,
    planner: str | Callable[[LLMCall], Any] = "",
    worker: str | Callable[[LLMCall], Any] = "",
    tester: str | Callable[[LLMCall], Any] = "",
) -> ScriptedCompletion:
    """Install a fresh :class:`ScriptedCompletion` with explicit replies."""
    scripted = ScriptedCompletion(planner=planner, worker=worker, tester=tester)
    monkeypatch.setattr(LLMClient, "complete", scripted)
    return scripted


# --------------------------------------------------------------------- factories


def make_settings(workspace: Path | str | None = None, **overrides: Any) -> Settings:
    """Build a :class:`Settings` isolated to *workspace* and free of network I/O."""
    kwargs: dict[str, Any] = {
        "workspace": Path(workspace) if workspace is not None else Path.cwd(),
        "retry_min_wait": 0.0,
        "test_timeout": 30,
    }
    kwargs.update(overrides)
    return Settings(**kwargs)


# ------------------------------------------------------------------ subprocesses


def write_pytest_shim(directory: Path) -> Path:
    """Write a ``pytest`` executable into *directory*; return *directory*.

    ``TesterAgent`` auto-detects the ``pytest`` *binary*; the sandbox only has
    ``python3``, so the binary is provided in-process instead of installing
    anything into the repository.
    """
    directory.mkdir(parents=True, exist_ok=True)
    shim = directory / "pytest"
    shim.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "from pytest import console_main\n"
        "sys.exit(console_main())\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return directory


def prepend_to_path(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    """Prepend *directory* to ``PATH`` for the duration of the test."""
    monkeypatch.setenv("PATH", f"{directory}{os.pathsep}{os.environ['PATH']}")


@pytest.fixture
def available_pytest_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make ``pytest`` resolvable as a bare command for the tester subprocess."""
    bindir = write_pytest_shim(tmp_path / "bin")
    prepend_to_path(monkeypatch, bindir)
    return bindir


def executable_shim(directory: Path, name: str, body: str) -> Path:
    """Write a small executable *name* into *directory* and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / name
    script.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    script.chmod(0o755)
    return script
