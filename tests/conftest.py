"""Shared fixtures for the furrow test suite.

The suite is fully offline. Two independent mechanisms guarantee that:

* :func:`_no_api_keys` (autouse) deletes every API key environment variable, so a
  stray real client construction raises ``ValueError: ANTHROPIC_API_KEY is not
  set`` instead of reaching the network;
* :class:`FakeLLM` replaces the provider SDKs with a table of canned replies, and
  :func:`stub_tester` stops the orchestrator from really shelling out to
  ``pytest -q`` / ``ruff check .`` / ``mypy .`` inside the workspace.

Helpers meant to be *used by* a test (``FakeLLM``, ``StubTesterRun``,
``plan_reply``, ``worker_reply``) are plain module-level names rather than
fixtures, so a test can import them or build them directly. The fixtures below
merely expose the same callables for tests that prefer to request them by name.
"""

from __future__ import annotations

import io
import json
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, ClassVar

import pytest
from rich.console import Console

import furrow.core.orchestrator
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    # ``typing.Self`` is 3.11+; the pydantic dependency ships the backport.
    from typing_extensions import Self

__all__ = [
    "DEFAULT_RESPONSE",
    "FakeLLM",
    "StubTesterRun",
    "plan_reply",
    "worker_reply",
]

# Environment variables that must never be visible to a test. ``Settings`` also
# reads ``FURROW_``-prefixed names, so those are cleared too.
API_KEY_ENV_VARS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "FURROW_ANTHROPIC_API_KEY",
    "FURROW_OPENAI_API_KEY",
    "OPENAI_API_KEY",
)

#: Reply used when a *role* model (planner/worker/tester) has no canned entry.
#: ``{}`` is valid JSON for every agent envelope: the planner builds an empty
#: plan, the worker reports a summary with no edits, the tester falls back to the
#: raw run result. See :meth:`FakeLLM.complete` for the exact rule.
DEFAULT_RESPONSE = "{}"

# Non-surprising settings defaults: the production ``max_parallel_tasks`` value,
# and a bounded cycle count so a test that forgets to bound ``Orchestrator.run()``
# still terminates. ``max_cycles=0`` means unlimited, which is never a sane test
# default; tests that need more cycles pass ``max_cycles=N`` explicitly.
DEFAULT_MAX_PARALLEL_TASKS = 5
DEFAULT_MAX_CYCLES = 3


@pytest.fixture(autouse=True)
def _no_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every API key from the environment for the duration of a test.

    Autouse, so no test can accidentally talk to a real provider: the SDK clients
    read ``os.getenv`` lazily, and a missing key makes them fail fast.
    """
    for name in API_KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class FakeLLM:
    """An offline stand-in for :class:`furrow.llm.LLMClient`.

    ``responses`` maps a **model name** to a canned reply. Dispatch happens inside
    :meth:`complete` from the ``model`` argument (falling back to
    ``settings.model``), which is what lets a single fake serve the planner, the
    workers and the tester, which use different models by default.

    A canned value may be:

    * a ``str`` -- returned verbatim;
    * an ``Exception``/``BaseException`` **instance** -- raised instead of
      returning. This is how a test drives a worker failure or a provider outage.

    Three attributes record what happened, for assertions:

    ``calls``
        ``[(model, prompt, system), ...]``, appended before the hook runs so the
        order reflects real execution order.
    ``writes``
        ``[(Path, content), ...]``, appended by :meth:`write_file`.
    ``hook``
        ``await hook(model=..., prompt=..., system=...)`` is invoked on every
        :meth:`complete`. Tests use it to observe ordering and concurrency (an
        ``asyncio.Event`` or an ``asyncio.Semaphore`` inside the hook is enough);
        :meth:`complete` never records the call *after* the hook, so a hook that
        blocks still shows up in ``calls`` immediately.

    File I/O is real (``read_file``/``write_file`` touch the filesystem and
    ``write_file`` creates parent directories), while :meth:`list_files` delegates
    to a real :class:`LLMClient` so pruning of ``__pycache__``/``.git``, sorting
    and ``limit`` stay faithful to production.
    """

    def __init__(
        self,
        responses: dict[str, object] | None = None,
        *,
        settings: Settings | None = None,
        workspace: Path | None = None,
        hook: Callable[..., Awaitable[None]] | None = None,
    ) -> None:
        self.settings: Settings = settings if settings is not None else Settings()
        self.responses: dict[str, object] = dict(responses or {})
        self.calls: list[tuple[str, str, str]] = []
        self.writes: list[tuple[Path, str]] = []
        self.hook: Callable[..., Awaitable[None]] | None = hook
        self.workspace: Path = (
            workspace if workspace is not None else Path(self.settings.workspace)
        )
        # Real client, used only for its filesystem listing helper. Constructing
        # it touches no network: the SDK clients are built lazily on first use and
        # :meth:`list_files` never uses them.
        self._real = LLMClient(settings=self.settings)

    # ---------------------------------------------------------------- helpers
    def _resolve_model(self, model: str | None) -> str:
        return model or self.settings.model

    def _role_models(self) -> set[str]:
        """Model names the agents actually dispatch on, per current settings."""
        return {
            str(self.settings.planner_model),
            str(self.settings.tester_model),
            str(self.settings.worker_model),
        }

    def _lookup(self, model: str) -> object:
        if model in self.responses:
            return self.responses[model]
        if model in self._role_models():
            # Explicitly documented fallback: a role model with no canned reply
            # yields a valid-but-empty JSON object instead of an exception, so a
            # partially wired test still exercises the parsing path. This IS a
            # silent default and can mask a mis-wired test, so it is deliberately
            # narrow: only the three role models qualify, while ``settings.model``
            # and any unknown name fail loudly.
            return DEFAULT_RESPONSE
        raise AssertionError(
            f"FakeLLM has no canned response for model {model!r}. "
            f"Registered models: {sorted(self.responses)!r}. "
            f"Settings role models (which fall back to {DEFAULT_RESPONSE!r}): "
            f"{sorted(self._role_models())!r}. "
            f"Register one with FakeLLM({{{model!r}: reply}}) or pass "
            f"settings= whose {model!r} model is registered."
        )

    # -------------------------------------------------------------- completion
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        """Return the canned reply for ``model``, or raise the canned exception.

        Raises ``AssertionError`` for a model that is neither registered nor one of
        the settings role models (see :meth:`_lookup`).
        """
        resolved = self._resolve_model(model)
        canned = self._lookup(resolved)
        self.calls.append((resolved, prompt, system))
        if self.hook is not None:
            await self.hook(model=resolved, prompt=prompt, system=system)
        if isinstance(canned, BaseException):
            raise canned
        if not isinstance(canned, str):
            return str(canned)
        return canned

    # --------------------------------------------------------------------- i/o
    async def read_file(self, path: str | Path) -> str:
        """Read ``path``, returning ``""`` when it does not exist."""
        target = Path(path)
        try:
            return target.read_text()
        except (OSError, ValueError):
            return ""

    async def write_file(self, path: str | Path, content: str) -> None:
        """Write ``content`` to ``path``, creating parent directories."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        self.writes.append((target, content))

    def list_files(
        self,
        directory: str | Path | None = None,
        ignore: Iterable[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """Delegate to a real :class:`LLMClient`, defaulting to ``self.workspace``."""
        root = self.workspace if directory is None else Path(directory)
        return self._real.list_files(root, ignore=ignore, limit=limit)

    # ---------------------------------------------------------------- lifecycle
    async def close(self) -> None:
        """No-op counterpart of :meth:`LLMClient.close`."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()


@pytest.fixture
def make_settings(tmp_path: Path) -> Callable[..., Settings]:
    """Return ``make_settings(**overrides) -> Settings`` bound to ``tmp_path``.

    No API keys, an isolated workspace, the production ``max_parallel_tasks`` and
    a bounded ``max_cycles`` (3). ``max_cycles=0`` means unlimited -- pass it
    explicitly only for a test that asserts the unbounded loop.
    """

    def _make_settings(**overrides: Any) -> Settings:
        values: dict[str, Any] = {
            "anthropic_api_key": None,
            "max_cycles": DEFAULT_MAX_CYCLES,
            "max_parallel_tasks": DEFAULT_MAX_PARALLEL_TASKS,
            "openai_api_key": None,
            "workspace": tmp_path,
        }
        values.update(overrides)
        return Settings(**values)

    return _make_settings


@pytest.fixture
def silent_console() -> Console:
    """A non-terminal :class:`rich.Console` writing into a throwaway buffer.

    Orchestrator output lands in ``console.file`` (a ``StringIO``), so it never
    pollutes captured stdout, and ``rich.Status`` renders nothing on a
    non-terminal console.
    """
    return Console(file=io.StringIO(), force_terminal=False, highlight=False, width=100)


@dataclass
class StubTesterRun:
    """One recorded ``StubTesterAgent.run`` invocation."""

    goal: str
    tasks: list[TaskModel]


@pytest.fixture
def stub_tester(monkeypatch: pytest.MonkeyPatch) -> Callable[..., type]:
    """Return ``stub_tester(results) -> type``, patching the orchestrator's tester.

    ``results`` is an ordered list of :class:`~furrow.config.TestResult` values;
    the last one repeats once the list is exhausted (an empty list is treated as a
    single passing result). The returned stub class is also
    ``furrow.core.orchestrator.TesterAgent`` for the duration of the test and
    exposes the recorded calls as ``stub.runs`` (a list of :class:`StubTesterRun`),
    so a test can assert exactly what goal and task list the orchestrator passed.

    **Mandatory for orchestrator tests**: without it ``_cycle()`` would really run
    ``pytest -q``, ``ruff check .`` and ``mypy .`` in the workspace.

    Usage::

        stub = stub_tester([TestResult(passed=True, summary="ok", failures=[])])
        ...
        assert stub.runs[0].goal == "ship it"
    """

    def _make_stub(results: Sequence[TestResult]) -> type:
        # The locals are named ``recorded_runs``/``scripted_results`` rather than
        # ``runs``/``scripted`` so that the class-body assignments below cannot
        # shadow them. A class body resolves its right-hand sides with
        # ``LOAD_NAME`` (class namespace -> module globals -> builtins) and never
        # sees the enclosing function scope, so a same-named local would raise
        # ``NameError`` at definition time. The class attributes are therefore
        # *annotated* in the body and *bound* after the ``class`` statement.
        recorded_runs: list[StubTesterRun] = []
        scripted_results: list[TestResult] = list(results) or [
            TestResult(passed=True, summary="stub tester (no results scripted)", failures=[])
        ]

        class StubTesterAgent:
            """Scripted replacement for :class:`furrow.agents.tester.TesterAgent`."""

            runs: ClassVar[list[StubTesterRun]]
            scripted: ClassVar[tuple[TestResult, ...]]
            instances: ClassVar[list[StubTesterAgent]]

            def __init__(
                self,
                client: Any = None,
                settings: Any = None,
                **kwargs: Any,
            ) -> None:
                self.client = client
                self.settings = settings
                self.kwargs = kwargs
                self.runs = recorded_runs
                StubTesterAgent.instances.append(self)

            async def run(
                self,
                goal: str,
                tasks: list[TaskModel],
                **kwargs: Any,
            ) -> TestResult:
                """Record the call and return the next scripted result."""
                recorded_runs.append(StubTesterRun(goal=goal, tasks=tasks))
                index = min(len(recorded_runs) - 1, len(scripted_results) - 1)
                return scripted_results[index]

        StubTesterAgent.runs = recorded_runs
        StubTesterAgent.scripted = tuple(scripted_results)
        StubTesterAgent.instances = []

        monkeypatch.setattr(furrow.core.orchestrator, "TesterAgent", StubTesterAgent)
        return StubTesterAgent

    return _make_stub


def plan_reply(*tasks: dict[str, object], rationale: str = "test plan") -> str:
    """Build a planner reply as a JSON string.

    The JSON is deliberately wrapped in a ```json fence behind a prose preamble,
    so the reply exercises the ``extract_json`` fence/prose stripping path rather
    than handing the parser a bare object. Each task mapping accepts the planner
    keys ``id``, ``description``, ``files`` and ``dependencies``.
    """
    payload: dict[str, object] = {
        "tasks": [dict(task) for task in tasks],
        "rationale": rationale,
    }
    body = json.dumps(payload, indent=2, sort_keys=True)
    return f"Here is the plan you asked for:\n\n```json\n{body}\n```\n"


def worker_reply(
    summary: str = "done",
    edits: Sequence[dict[str, str]] | None = None,
    issues: Sequence[str] | None = None,
) -> str:
    """Build a worker reply as a plain JSON string.

    ``edits`` entries are ``{"path": ..., "content": ...}`` full-file writes;
    leaving ``edits`` empty yields a summary-only success. Any ``issues`` makes the
    worker report them alongside (and, with no written edits, fail the task).
    """
    payload: dict[str, object] = {
        "edits": [dict(edit) for edit in (edits or ())],
        "issues": [str(issue) for issue in (issues or ())],
        "summary": summary,
    }
    return json.dumps(payload)


@pytest.fixture(name="plan_reply")
def _plan_reply_fixture() -> Callable[..., str]:
    """Expose :func:`plan_reply` as a fixture (no arguments)."""
    return plan_reply


@pytest.fixture(name="worker_reply")
def _worker_reply_fixture() -> Callable[..., str]:
    """Expose :func:`worker_reply` as a fixture (no arguments)."""
    return worker_reply