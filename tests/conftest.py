"""Shared fixtures and offline fakes for the furrow regression suite.

``pyproject.toml`` sets ``asyncio_mode = "auto"``, but ``pytest-asyncio`` is **not**
installed in this environment, so that option is inert. Every async test in this suite is
therefore written as a *synchronous* function that drives an inner coroutine through
:func:`asyncio.run` -- see ``tests/test_orchestrator.py`` for the pattern.

Nothing here touches the network: :class:`StubLLMClient` answers from canned queues and
the real subprocess test-runner candidates are replaced with a one-line Python script.
"""

from __future__ import annotations

import asyncio
import importlib
import io
import json
import os
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from furrow.config import Settings
from furrow.config import settings as singleton_settings
from furrow.llm import LLMClient

#: Distinguishing fragments of the three agent prompts, so a stub can tell them apart.
PLANNER_MARKER = "You are a planning agent"
WORKER_MARKER = "You are a worker agent"
TESTER_MARKER = "Given the goal and test output"

#: A test runner that exists on every machine but is instant and offline.
STUB_RUNNER = (sys.executable, "-c", "print('furrow test runner stub')")

#: ``TEST_COMMANDS`` exactly as shipped, captured before the autouse fixture below replaces
#: it. Lets a test assert on the real candidate list despite the stub.
REAL_TEST_COMMANDS = importlib.import_module("furrow.agents.tester").TEST_COMMANDS

PASSING_VERDICT = '{"passed": true, "summary": "all tests passed", "failures": []}'
FAILING_VERDICT = (
    '{"passed": false, "summary": "1 failed, 1 passed", "failures": '
    '["tests/test_widget.py::test_build AssertionError: expected 3"]}'
)


def task_json(
    task_id: str,
    description: str,
    files: Iterable[str] = (),
    dependencies: Iterable[str] = (),
) -> dict[str, Any]:
    """Build one raw planner task entry."""
    return {
        "id": task_id,
        "description": description,
        "files": list(files),
        "dependencies": list(dependencies),
    }


def plan_json(*tasks: dict[str, Any], rationale: str = "because") -> str:
    """Serialise a whole plan the way a well-behaved model would."""
    return json.dumps({"tasks": list(tasks), "rationale": rationale})


#: What a stub planner falls back to once its ``plans`` queue runs dry. One real task, because
#: ``PlannerAgent`` rejects an empty task list and the orchestrator halts on that.
DEFAULT_PLAN = plan_json(task_json("a", "Do A"))


def task_label(prompt: str) -> str:
    """Pull the ``Task: ...`` line back out of a worker prompt."""
    for line in prompt.splitlines():
        if line.startswith("Task: "):
            return line.removeprefix("Task: ")
    return prompt


@dataclass(frozen=True)
class Call:
    """One recorded call into :meth:`StubLLMClient.complete`."""

    prompt: str
    system: str
    model: str | None


class StubLLMClient(LLMClient):
    """An :class:`LLMClient` that replays canned answers instead of calling a provider.

    The three agents each pass a distinct prompt, so the stub routes on the prompt body:
    planner prompts drain ``plans``, tester prompts drain ``verdicts`` and worker prompts
    drain ``summaries`` (or whatever ``worker_hook`` returns). When a queue runs dry the
    last entry is repeated, which keeps long orchestrator runs deterministic.

    ``worker_yields`` inserts ``await asyncio.sleep(0)`` hops at the start of every worker
    call. That is what makes real concurrency observable: every task in a batch reaches its
    ``await`` before any of them finishes, so :attr:`peak_concurrency` reflects the true
    simultaneous fan-out rather than an artefact of scheduling.

    With no ``settings=`` argument the stub holds the module-level singleton, exactly as a real
    ``LLMClient()`` does. That is what lets a test pin ``max_cycles``/``max_parallel_tasks`` on
    the global (see :func:`tests.test_orchestrator.limit_cycles` and
    :func:`unlimited_cycles` below) and have the run actually honour it, whatever object the
    orchestrator reads its limits from. Pass ``settings=Settings(...)`` explicitly for a run
    that must be configured independently of the global.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        plans: Iterable[str] = (),
        verdicts: Iterable[str] = (),
        summaries: Iterable[str] = (),
        worker_hook: Callable[[str], str | BaseException | None] | None = None,
        worker_yields: int = 5,
        planner_error: BaseException | None = None,
        tester_error: BaseException | None = None,
    ) -> None:
        super().__init__(settings=settings if settings is not None else singleton_settings)
        self.plans = list(plans)
        self.verdicts = list(verdicts)
        self.summaries = list(summaries)
        self.worker_hook = worker_hook
        self.worker_yields = worker_yields
        self.planner_error = planner_error
        self.tester_error = tester_error
        self.calls: list[Call] = []
        self.planner_prompts: list[str] = []
        self.worker_prompts: list[str] = []
        self.tester_prompts: list[str] = []
        self.worker_order: list[str] = []
        self.active_workers = 0
        self.peak_concurrency = 0

    @staticmethod
    def _next(queue: list[str], fallback: str) -> str:
        return queue.pop(0) if queue else fallback

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.calls.append(Call(prompt, system, model))
        if PLANNER_MARKER in prompt:
            self.planner_prompts.append(prompt)
            if self.planner_error is not None:
                raise self.planner_error
            return self._next(self.plans, DEFAULT_PLAN)
        if TESTER_MARKER in prompt:
            self.tester_prompts.append(prompt)
            if self.tester_error is not None:
                raise self.tester_error
            return self._next(self.verdicts, PASSING_VERDICT)
        return await self._run_worker(prompt)

    async def _run_worker(self, prompt: str) -> str:
        label = task_label(prompt)
        self.worker_prompts.append(prompt)
        self.worker_order.append(label)
        self.active_workers += 1
        self.peak_concurrency = max(self.peak_concurrency, self.active_workers)
        try:
            for _ in range(self.worker_yields):
                # Yield to the event loop so sibling tasks in the same batch interleave and
                # the semaphore in Orchestrator._execute is genuinely exercised.
                await asyncio.sleep(0)
            if self.worker_hook is not None:
                outcome = self.worker_hook(prompt)
                if isinstance(outcome, BaseException):
                    raise outcome
                if outcome is not None:
                    return outcome
            return self._next(self.summaries, f"summary for {label}")
        finally:
            self.active_workers -= 1


@pytest.fixture
def make_client() -> Callable[..., StubLLMClient]:
    """Factory for :class:`StubLLMClient`, by default bound to the global settings singleton."""

    def _make(**kwargs: Any) -> StubLLMClient:
        return StubLLMClient(**kwargs)

    return _make


@pytest.fixture
def console_output(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Redirect the orchestrator's rich console into a buffer.

    Keeps the test log readable and lets a test assert on console output if it wants to.
    It does not weaken anything: rich still parses markup, so a ``MarkupError`` regression
    would surface exactly the same way with a real terminal attached.
    """
    buffer = io.StringIO()
    orchestrator = importlib.import_module("furrow.core.orchestrator")
    monkeypatch.setattr(orchestrator, "console", Console(file=buffer, width=200))
    return buffer


@pytest.fixture(autouse=True)
def stub_test_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop every test from shelling out to the repository's own pytest.

    ``TesterAgent.__init__`` copies ``TEST_COMMANDS`` into ``self.test_commands`` when it is
    constructed, and the orchestrator constructs it per cycle, so patching the module
    attribute here is enough to redirect every runner. Tests that care about runner
    selection override ``agent.test_commands`` directly.
    """
    tester = importlib.import_module("furrow.agents.tester")
    monkeypatch.setattr(tester, "TEST_COMMANDS", (STUB_RUNNER,))


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Strip ambient configuration so tests cannot be polluted by the host.

    ``Settings`` reads ``FURROW_``-prefixed environment variables *and* a ``.env`` file from
    the current directory, so both sources are neutralised: every ``FURROW_*`` variable (plus
    the unprefixed provider keys) is removed and the working directory is moved into
    ``tmp_path`` where no ``.env`` exists. Nothing in the suite may write to the repo tree.
    """
    for name in list(os.environ):
        if name.startswith("FURROW_") or name in {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"}:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def repo_root() -> Path:
    """The repository root, derived from this file's location (not from ``os.getcwd()``)."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def restore_settings_singleton(monkeypatch: pytest.MonkeyPatch) -> Iterable[None]:
    """Restore ``furrow.config.settings`` after every test.

    The orchestrator, the CLI and the web endpoint all mutate the module-level singleton in
    place, so without this a CLI test would leave ``max_cycles`` pinned for the whole run.

    The ordering against ``monkeypatch`` is made explicit instead of being left to fixture
    teardown order. pytest finalises the fixture that was set up *last* first, and
    ``monkeypatch`` is set up before this one (``isolate_environment`` requests it), so this
    teardown otherwise runs *before* the undo and the undo wins -- correct today only because
    both write the same value. Undoing first and restoring second makes the guarantee hold
    whatever the order is; the automatic undo afterwards is a documented no-op.
    """
    snapshot = {name: getattr(singleton_settings, name) for name in Settings.model_fields}
    yield
    monkeypatch.undo()
    for name, value in snapshot.items():
        setattr(singleton_settings, name, value)


@pytest.fixture
def unlimited_cycles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee the ``max_cycles=0`` (unlimited) configuration for one test."""
    monkeypatch.setattr(singleton_settings, "max_cycles", 0)
