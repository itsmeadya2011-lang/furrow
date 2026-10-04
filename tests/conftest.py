"""Shared, hermetic fixtures for the Furrow test suite.

Nothing here touches the network, spawns a provider SDK, sleeps for real, or writes outside
pytest's ``tmp_path``. Fixtures are deliberately well named and reusable: the agent that owns
``tests/test_web.py`` and ``tests/test_cli.py`` should import these instead of rebuilding them.

Fixtures
--------
``tmp_workspace``      empty directory used as ``Settings.workspace``
``settings``           real ``Settings`` pointed at ``tmp_workspace``
``make_settings``      factory for ``Settings`` with overrides
``fake_llm``           ``FakeLLMClient`` replaying one canned response forever
``make_fake_llm``      factory: ``make_fake_llm(responses=[...], max_cycles=2)``
``stub_agents``        installs ``PlannerStub``/``WorkerStub``/``TesterStub`` into the loop
``event_recorder``     ``Orchestrator(on_event=...)`` handler that records every event
``captured_console``   redirects the orchestrator's rich console into an in-memory buffer
``bounded``            runs an awaitable under a hard deadline (no pytest-timeout needed)
``subprocess_stub``    scripted replacement for ``asyncio.create_subprocess_exec``
``recorded_sleeps``    makes ``asyncio.sleep`` instant while recording the requested delays

Two autouse fixtures keep the process-wide ``furrow.config`` settings cache from leaking
between tests: ``isolated_settings`` and ``provider_api_keys``.

Importable helpers (also usable without a fixture): ``build_settings``, ``make_plan``,
``passing``, ``failing``, plus the doubles ``FakeLLMClient``, ``EventRecorder``, ``SDKStub``
(in ``test_llm.py``), ``PlannerStub``, ``WorkerStub``, ``TesterStub``, ``SubprocessStub`` and
``ProcessOutcome``.

Note on deadlines: ``Orchestrator.run`` turns a cancellation into a ``LoopResult`` with
``stopped_reason="cancelled"`` instead of propagating it, so wrapping it in ``bounded`` reports a
deadlock as a cancelled result rather than a ``TimeoutError``. Assert on the reason when you
expect the loop to finish.
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import pytest
from rich.console import Console

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    # An editable install of "furrow" from another checkout lives in site-packages; keeping this
    # repository first guarantees the tests exercise the code in this working tree.
    sys.path.insert(0, str(REPO_ROOT))

import furrow  # noqa: E402  (path bootstrap above must run first)
from furrow.agents.tester import TesterAgent  # noqa: E402
from furrow.config import (  # noqa: E402
    Plan,
    Settings,
    TaskModel,
    TestResult,
    reset_settings,
)
from furrow.core import orchestrator as orchestrator_module  # noqa: E402
from furrow.llm import LLMClient  # noqa: E402

# These two names start with "Test", so pytest tries to collect them as test classes and warns.
# This is the documented opt-out; it cannot live in furrow/** (which this session does not own),
# so it is set once here for the whole suite.
PYTEST_COLLECT_FLAG = "__test__"
setattr(TestResult, PYTEST_COLLECT_FLAG, False)
setattr(TesterAgent, PYTEST_COLLECT_FLAG, False)

T = TypeVar("T")

#: Default deadline for scheduler tests: generous for CI, short enough that a deadlock fails.
DEADLINE_SECONDS = 10.0


def build_settings(**overrides: Any) -> Settings:
    """Build real :class:`Settings` with ``.env`` loading disabled.

    ``.env`` is skipped so a developer's local file can never change what a test sees; pass
    ``_env_file=...`` explicitly in the rare test that exercises dotenv loading.
    """
    values: dict[str, Any] = {"_env_file": None}
    values.update(overrides)
    return Settings(**values)


def assert_repo_furrow() -> None:
    """Fail loudly if a foreign ``furrow`` distribution shadows this checkout."""
    imported = Path(furrow.__file__).resolve().parent
    if imported != REPO_ROOT / "furrow":
        raise ImportError(
            f"tests must import furrow from {REPO_ROOT}, but got {imported}; "
            "a stale editable install is shadowing the working tree"
        )


assert_repo_furrow()


# --------------------------------------------------------------------------------------
# Fake LLM client
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class LLMCall:
    """One recorded call to :meth:`FakeLLMClient.complete`."""

    prompt: str
    system: str
    model: str | None


class FakeLLMClient(LLMClient):
    """An :class:`~furrow.llm.LLMClient` whose only stubbed method is ``complete``.

    ``read_file``/``write_file``/``list_files`` are inherited, so the real (aiofiles-backed)
    implementations run against a real temporary workspace. No provider SDK is ever constructed,
    so a test can never make a network call by accident.

    ``responses`` may be a single string (replayed for every call) or a sequence consumed in
    order; with ``repeat_last=True`` (the default) the final item is replayed once the queue is
    exhausted, which is what a multi-cycle loop usually wants. An item may also be an exception
    instance (alone or inside the sequence), which is raised instead of returned.
    """

    def __init__(
        self,
        settings: Settings,
        responses: str | Sequence[str | BaseException] = "ok",
        repeat_last: bool = True,
    ) -> None:
        super().__init__(settings=settings)
        self.calls: list[LLMCall] = []
        self.repeat_last = repeat_last
        if isinstance(responses, (str, BaseException)):
            self._queue: list[str | BaseException] = [responses]
        else:
            self._queue = list(responses)

    def queue(self, *responses: str | BaseException) -> FakeLLMClient:
        """Append more scripted outcomes and return self, for fluent test setup."""
        self._queue.extend(responses)
        return self

    @property
    def prompts(self) -> list[str]:
        return [call.prompt for call in self.calls]

    @property
    def systems(self) -> list[str]:
        return [call.system for call in self.calls]

    @property
    def models(self) -> list[str | None]:
        return [call.model for call in self.calls]

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def last_call(self) -> LLMCall:
        assert self.calls, "FakeLLMClient was never called"
        return self.calls[-1]

    def _next(self) -> str | BaseException:
        if not self._queue:
            raise AssertionError("FakeLLMClient ran out of scripted responses")
        if len(self._queue) == 1:
            return self._queue[0] if self.repeat_last else self._queue.pop()
        return self._queue.pop(0)

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.calls.append(LLMCall(prompt=prompt, system=system, model=model))
        outcome = self._next()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


# --------------------------------------------------------------------------------------
# Loop doubles (planner / worker / tester)
# --------------------------------------------------------------------------------------


@dataclass
class PlannerStub:
    """Replays scripted plans (or raises scripted exceptions) for the loop."""

    outcomes: Sequence[Plan | BaseException] = ()
    goals: list[str] = field(default_factory=list)
    calls: int = 0

    async def plan(self, goal: str) -> Plan:
        self.goals.append(goal)
        self.calls += 1
        if not self.outcomes:
            raise AssertionError("PlannerStub was given no plans to replay")
        index = min(self.calls - 1, len(self.outcomes) - 1)
        outcome = self.outcomes[index]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@dataclass
class WorkerStub:
    """Records the tasks handed to a worker and scripts each task's outcome.

    ``results`` maps a task id to a summary string or an exception instance. ``on_run`` is an
    optional async hook awaited *before* the result is produced, which is how a test observes
    concurrency or parks a worker until it is cancelled.
    """

    results: dict[str, str | BaseException] = field(default_factory=dict)
    default_result: str = "completed"
    on_run: Callable[[TaskModel], Awaitable[None]] | None = None
    calls: list[TaskModel] = field(default_factory=list)

    @property
    def started_ids(self) -> list[str]:
        return [task.id for task in self.calls]

    async def run(self, task: TaskModel) -> str:
        self.calls.append(task)
        if self.on_run is not None:
            await self.on_run(task)
        outcome = self.results.get(task.id, self.default_result)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@dataclass
class TesterStub:
    """Replays scripted test verdicts for the loop."""

    outcomes: Sequence[TestResult | BaseException] = ()
    goals: list[str] = field(default_factory=list)
    task_batches: list[list[TaskModel]] = field(default_factory=list)
    calls: int = 0

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        self.goals.append(goal)
        self.task_batches.append(list(tasks))
        self.calls += 1
        if not self.outcomes:
            raise AssertionError("TesterStub was given no verdicts to replay")
        index = min(self.calls - 1, len(self.outcomes) - 1)
        outcome = self.outcomes[index]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@dataclass
class AgentStubs:
    """The three loop doubles installed by the ``stub_agents`` fixture."""

    planner: PlannerStub
    worker: WorkerStub
    tester: TesterStub


# --------------------------------------------------------------------------------------
# Subprocess double
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class ProcessOutcome:
    """What a stubbed child process should do when it is created."""

    returncode: int | None = 0
    stdout: bytes = b""
    stderr: bytes = b""
    hang: bool = False

    @classmethod
    def of(
        cls,
        returncode: int | None = 0,
        *,
        stdout: str = "",
        stderr: str = "",
        hang: bool = False,
    ) -> ProcessOutcome:
        return cls(
            returncode=returncode,
            stdout=stdout.encode(),
            stderr=stderr.encode(),
            hang=hang,
        )


class FakeProcess:
    """Minimal stand-in for ``asyncio.subprocess.Process``."""

    def __init__(self, outcome: ProcessOutcome) -> None:
        self.returncode: int | None = outcome.returncode
        self._stdout = outcome.stdout
        self._stderr = outcome.stderr
        self._hang = outcome.hang
        self.killed = False
        self.reaped = False
        self.release = asyncio.Event()

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._hang:
            await self.release.wait()
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True
        if self.returncode is None:
            self.returncode = -9
        self.release.set()

    async def wait(self) -> int:
        self.reaped = True
        return self.returncode if self.returncode is not None else -9


class SubprocessStub:
    """Callable installed over ``asyncio.create_subprocess_exec``.

    Records every command it was asked to spawn and hands back a :class:`FakeProcess` built from
    the next scripted :class:`ProcessOutcome` (the last one repeats).
    """

    def __init__(self, *outcomes: ProcessOutcome) -> None:
        self._queue: list[ProcessOutcome] = list(outcomes)
        self._default = ProcessOutcome()
        self.commands: list[tuple[str, ...]] = []
        self.kwargs: list[dict[str, Any]] = []
        self.processes: list[FakeProcess] = []

    def configure(self, **changes: Any) -> SubprocessStub:
        """Update the fallback outcome used once the queue is drained."""
        for key, value in changes.items():
            if key in ("stdout", "stderr") and isinstance(value, str):
                value = value.encode()
            setattr(self._default, key, value)
        return self

    def queue(self, *outcomes: ProcessOutcome) -> SubprocessStub:
        self._queue.extend(outcomes)
        return self

    def _next(self) -> ProcessOutcome:
        if not self._queue:
            return self._default
        return self._queue.pop(0) if len(self._queue) > 1 else self._queue[0]

    async def __call__(self, *command: str, **kwargs: Any) -> FakeProcess:
        self.commands.append(tuple(command))
        self.kwargs.append(kwargs)
        process = FakeProcess(self._next())
        self.processes.append(process)
        return process

    @property
    def command(self) -> tuple[str, ...]:
        assert self.commands, "no subprocess was spawned"
        return self.commands[0]

    @property
    def process(self) -> FakeProcess:
        assert self.processes, "no subprocess was spawned"
        return self.processes[0]


# --------------------------------------------------------------------------------------
# Event recorder
# --------------------------------------------------------------------------------------


class EventRecorder:
    """A sync ``Orchestrator(on_event=...)`` handler that keeps every event it saw."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, name: str, payload: dict[str, Any]) -> None:
        self.events.append((name, dict(payload)))

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.events]

    def payloads(self, name: str) -> list[dict[str, Any]]:
        return [payload for event, payload in self.events if event == name]

    def count(self, name: str) -> int:
        return self.names.count(name)

    def messages(self, name: str) -> list[str]:
        return [str(payload.get("message")) for payload in self.payloads(name)]


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep ``furrow.config``'s process-wide settings cache from leaking between tests.

    Runs before and after every test so a ``FURROW_*`` override, or an override applied through
    ``get_settings(**overrides)``, can never reach the next test.
    """
    for name in [key for key in os.environ if key.startswith("FURROW_")]:
        monkeypatch.delenv(name, raising=False)
    reset_settings()
    try:
        yield
    finally:
        reset_settings()


@pytest.fixture(autouse=True)
def provider_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop ambient provider keys so no test can accidentally authenticate."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """An empty directory to use as ``Settings.workspace``."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def make_settings(tmp_workspace: Path) -> Callable[..., Settings]:
    """Build real :class:`Settings` rooted at ``tmp_workspace``.

    ``.env`` loading is disabled (``_env_file=None``) so a developer's local ``.env`` can never
    change what a test sees. ``max_retries=1`` keeps an unexpected retry path from sleeping.
    """

    def _make(**overrides: Any) -> Settings:
        values: dict[str, Any] = {
            "workspace": tmp_workspace,
            "anthropic_api_key": "test-anthropic-key",
            "openai_api_key": "test-openai-key",
            "max_retries": 1,
            "test_timeout": 5.0,
        }
        values.update(overrides)
        return Settings(**values)

    return _make


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    """The default :class:`Settings` used by most tests."""
    return make_settings()


@pytest.fixture
def make_fake_llm(
    make_settings: Callable[..., Settings],
) -> Callable[..., FakeLLMClient]:
    """Build a :class:`FakeLLMClient`, optionally overriding settings.

    ``make_fake_llm(responses=[...], max_cycles=2, max_parallel_tasks=2)``
    """

    def _make(
        responses: str | Sequence[str | BaseException] = "ok",
        **setting_overrides: Any,
    ) -> FakeLLMClient:
        return FakeLLMClient(make_settings(**setting_overrides), responses=responses)

    return _make


@pytest.fixture
def fake_llm(make_fake_llm: Callable[..., FakeLLMClient]) -> FakeLLMClient:
    """A fake client that answers every call with ``"ok"``."""
    return make_fake_llm("ok")


@pytest.fixture
def stub_agents(monkeypatch: pytest.MonkeyPatch) -> Callable[..., AgentStubs]:
    """Replace the loop's three agents with recording doubles (no LLM, no subprocess).

    The doubles are installed as classes because ``Orchestrator`` constructs a fresh
    ``WorkerAgent``/``TesterAgent`` per task/cycle; they only need to accept the same keyword
    arguments. Returns an :class:`AgentStubs` handle for assertions.
    """

    def _install(
        *,
        plans: Sequence[Plan | BaseException] = (),
        test_results: Sequence[TestResult | BaseException] = (),
        worker_results: dict[str, str | BaseException] | None = None,
        worker_hook: Callable[[TaskModel], Awaitable[None]] | None = None,
        default_worker_result: str = "completed",
    ) -> AgentStubs:
        stubs = AgentStubs(
            planner=PlannerStub(outcomes=plans),
            worker=WorkerStub(
                results=worker_results or {},
                default_result=default_worker_result,
                on_run=worker_hook,
            ),
            tester=TesterStub(outcomes=test_results),
        )

        class _Planner:
            def __init__(self, client: Any = None, settings: Any = None) -> None:
                self.client = client

            async def plan(self, goal: str) -> Plan:
                return await stubs.planner.plan(goal)

        class _Worker:
            def __init__(self, task: Any, client: Any = None, settings: Any = None) -> None:
                self.task = task

            async def run(self) -> str:
                return await stubs.worker.run(self.task)

        class _Tester:
            def __init__(self, client: Any = None, settings: Any = None) -> None:
                self.client = client

            async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
                return await stubs.tester.run(goal, tasks)

        monkeypatch.setattr(orchestrator_module, "PlannerAgent", _Planner)
        monkeypatch.setattr(orchestrator_module, "WorkerAgent", _Worker)
        monkeypatch.setattr(orchestrator_module, "TesterAgent", _Tester)
        return stubs

    return _install


@pytest.fixture
def event_recorder() -> EventRecorder:
    """A pass to ``Orchestrator(on_event=...)`` that records every event."""
    return EventRecorder()


@pytest.fixture
def captured_console() -> Iterator[io.StringIO]:
    """Capture the orchestrator's rich console into a buffer.

    Opt-in rather than autouse so CLI tests can keep asserting on real stdout. The orchestrator
    reads the module global at call time, so patching it here redirects the banner, per-cycle
    output and ``Status`` spinners.
    """
    buffer = io.StringIO()
    original = orchestrator_module.console
    orchestrator_module.console = Console(
        file=buffer, width=200, no_color=True, highlight=False, soft_wrap=True
    )
    try:
        yield buffer
    finally:
        orchestrator_module.console = original


@pytest.fixture
def bounded() -> Callable[..., Awaitable[Any]]:
    """Return a runner that enforces a hard deadline around an awaitable.

    ``await bounded(orchestrator.run())`` turns a scheduler deadlock into a finished-but-cancelled
    ``LoopResult`` instead of a hung suite; ``pytest-timeout`` is not a dependency. Check
    ``result.stopped_reason`` when the loop is supposed to complete.
    """

    def _bounded(awaitable: Awaitable[T], timeout: float = DEADLINE_SECONDS) -> Awaitable[T]:
        return asyncio.wait_for(awaitable, timeout)

    return _bounded


@pytest.fixture
def subprocess_stub(monkeypatch: pytest.MonkeyPatch) -> SubprocessStub:
    """Replace ``asyncio.create_subprocess_exec`` with a scripted stub."""
    stub = SubprocessStub()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", stub)
    return stub


@pytest.fixture
def recorded_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Make ``asyncio.sleep`` instant while recording the delays it was asked for.

    tenacity resolves ``asyncio.sleep`` at call time, so patching it here keeps the LLM retry
    backoff honest (the test can assert the delays) without paying for them.
    """
    real_sleep = asyncio.sleep
    delays: list[float] = []

    async def instant_sleep(delay: float, *args: object, **kwargs: object) -> None:
        delays.append(float(delay))
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)
    return delays


# --------------------------------------------------------------------------------------
# Small helpers shared by several test modules
# --------------------------------------------------------------------------------------


def make_plan(*tasks: str, rationale: str = "because") -> Plan:
    """Build a ``Plan`` whose tasks are named with positional ids ``1``, ``2``, ..."""
    return Plan(
        tasks=[TaskModel(id=str(index), description=task) for index, task in enumerate(tasks, 1)],
        rationale=rationale,
    )


def passing(summary: str = "all green") -> TestResult:
    """A passing ``TestResult``."""
    return TestResult(passed=True, summary=summary)


def failing(*failures: str, summary: str = "boom") -> TestResult:
    """A failing ``TestResult`` carrying the given failure lines."""
    return TestResult(passed=False, summary=summary, failures=list(failures))
