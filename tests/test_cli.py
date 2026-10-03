"""Command-line regression tests for :mod:`furrow.cli.main`.

What is protected here:

* Every command answers ``--help`` with exit 0, ``version`` reports the Furrow and
  Python versions without needing an API key, and usage errors exit 2 with no
  traceback on stderr.
* **The global-settings regression:** ``_build_settings`` folds CLI overrides into
  a per-invocation copy and never mutates the ``furrow.config.settings`` singleton.
  The old implementation did ``settings.model = model``, which leaked the override
  into every later command; these tests pin the singleton byte-for-byte while also
  asserting the overrides really did reach the orchestrator and the planner.
* Option to field mapping (``--cycles``, ``--max-parallel``, ``--workspace``,
  ``--test-command``, ``--test-timeout``), mixed-case ``--provider``, and the fact
  that options the user did not pass leave those fields alone.
* ``plan`` renders the plan table without executing anything, ``start`` prompts for
  a missing goal via ``click.prompt``, and a ``KeyboardInterrupt`` mid-run exits
  with the documented code instead of unwinding a traceback.

No loop ever runs here: the orchestrator and the planner are replaced with offline
doubles, so nothing touches an LLM, a workspace or a subprocess.
"""

from __future__ import annotations

import inspect
import platform
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
import pytest
from click.testing import CliRunner

from furrow.cli import main as cli_module
from furrow.config import Plan, Provider, TaskModel
from furrow.config import settings as global_settings
from furrow.core import Event, EventType

#: The click group under test.
main = cli_module.main
#: The exit code the CLI documents for an aborted run.
EXIT_INTERRUPTED = cli_module.EXIT_INTERRUPTED
_build_settings = cli_module._build_settings
_package_version = cli_module._package_version

#: ``furrow <version> (python <version>)``.
_VERSION_LINE = re.compile(r"furrow (?P<furrow>\S+) \(python (?P<python>[\d.]+)\)")

#: Every option the CLI folds into a settings field, as ``option -> field``.
SETTING_OPTIONS: tuple[tuple[str, str], ...] = (
    ("--cycles", "max_cycles"),
    ("--max-parallel", "max_parallel_tasks"),
    ("--workspace", "workspace"),
    ("--test-command", "test_command"),
    ("--test-timeout", "test_timeout"),
)


# --------------------------------------------------------------------- doubles


class RecordingOrchestrator:
    """Offline stand-in for :class:`furrow.core.Orchestrator`.

    It records the goal and the settings object the CLI built for it, and its
    ``run`` returns immediately, so a command finishes in milliseconds without a
    planner, an LLM or a workspace. ``raise_on_run`` makes it blow up instead,
    which is how the interrupt path is exercised.
    """

    def __init__(
        self,
        goal: str,
        client: Any = None,
        settings: Any = None,
        on_event: Callable[[Any], Any] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client
        self.settings = settings
        self.on_event = on_event
        self.stop_requested = False
        self.stop_during_run = False
        self.raise_on_run: BaseException | None = None
        self.max_cycles: int | None = None

    def request_stop(self) -> None:
        """Record that something asked the run to stop."""
        self.stop_requested = True

    def state(self) -> dict[str, Any]:
        """Return a snapshot; ``done`` is false only when the run was stopped."""
        return {
            "goal": self.goal,
            "cycles": 1,
            "done": not self.stop_during_run,
            "tasks": [],
            "history": [],
        }

    async def run(self, max_cycles: int | None = None) -> None:
        """Return at once, optionally stopping or raising on the way out."""
        if self.raise_on_run is not None:
            raise self.raise_on_run
        self.max_cycles = max_cycles
        if self.stop_during_run:
            self.request_stop()
        await self._emit_done()

    async def _emit_done(self) -> None:
        """Emit the closing ``DONE`` frame the real loop always ends with.

        The renderer only learns a run was stopped from this frame's
        ``stop_requested`` flag, so a double that stayed silent would not exercise
        the stopped-run exit path at all.
        """
        if self.on_event is None:
            return
        event = Event(
            type=EventType.DONE,
            cycle=1,
            message="Stop requested by caller." if self.stop_during_run else "Goal complete.",
            data={
                "goal": self.goal,
                "cycles": 1,
                "done": not self.stop_during_run,
                "stop_requested": self.stop_requested,
            },
        )
        outcome = self.on_event(event)
        if inspect.isawaitable(outcome):
            await outcome


class OrchestratorDouble:
    """Callable installed as ``cli_module.Orchestrator``; collects what it built."""

    def __init__(self) -> None:
        self.built: list[RecordingOrchestrator] = []
        self.raise_on_run: BaseException | None = None
        self.stop_during_run = False

    def __call__(
        self,
        goal: str,
        client: Any = None,
        settings: Any = None,
        on_event: Callable[[Any], Any] | None = None,
    ) -> RecordingOrchestrator:
        """Build one recording double for the command under test."""
        orchestrator = RecordingOrchestrator(
            goal, client=client, settings=settings, on_event=on_event
        )
        orchestrator.raise_on_run = self.raise_on_run
        orchestrator.stop_during_run = self.stop_during_run
        self.built.append(orchestrator)
        return orchestrator

    @property
    def only(self) -> RecordingOrchestrator:
        """Return the single orchestrator that was built, failing otherwise."""
        assert len(self.built) == 1, f"expected one run, built {len(self.built)}"
        return self.built[0]


class SingletonGuard:
    """Snapshots ``furrow.config.settings`` so a test can prove it stayed untouched."""

    def __init__(self) -> None:
        self.before = global_settings.model_dump_json()

    def assert_unchanged(self) -> None:
        """Fail if any code path rewrote the process-wide settings singleton."""
        after = global_settings.model_dump_json()
        assert after == self.before, (
            "furrow.config.settings was mutated by the CLI\n"
            f"before: {self.before}\nafter:  {after}"
        )


# --------------------------------------------------------------------- fixtures


@pytest.fixture
def runner() -> CliRunner:
    """A fresh in-process click runner (no subprocess, no shell)."""
    return CliRunner()


@pytest.fixture
def singleton() -> SingletonGuard:
    """Snapshot the global settings singleton before the test body runs."""
    return SingletonGuard()


@pytest.fixture
def orchestrator(monkeypatch: pytest.MonkeyPatch) -> OrchestratorDouble:
    """Replace the orchestrator so no loop, LLM or workspace is ever touched."""
    double = OrchestratorDouble()
    monkeypatch.setattr(cli_module, "Orchestrator", double)
    return double


@pytest.fixture
def planner(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the planner with one canned task; returns the recorded calls."""
    calls: list[dict[str, Any]] = []

    class PlannerDouble:
        """Offline planner returning a single task."""

        def __init__(self, client: Any = None, settings: Any = None) -> None:
            self.client = client
            self.settings = settings
            calls.append({"settings": settings})

        async def plan(self, goal: str, context: str | None = None) -> Plan:
            """Record the call and hand back a fixed one-task plan."""
            calls[-1].update(goal=goal, context=context)
            return Plan(
                tasks=[TaskModel(id="task-1", description="add the tests")],
                rationale="the goal needs tests",
            )

    monkeypatch.setattr("furrow.agents.planner.PlannerAgent", PlannerDouble)
    return calls


# ---------------------------------------------------------------------- help


@pytest.mark.parametrize(
    "args",
    [["--help"], ["start", "--help"], ["plan", "--help"], ["web", "--help"], ["version", "--help"]],
)
def test_help_exits_zero_for_the_group_and_every_command(runner: CliRunner, args: list[str]) -> None:
    """Every command documents itself and answers ``--help`` with exit 0."""
    result = runner.invoke(main, args)

    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_version_reports_both_versions_without_an_api_key(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``version`` needs no credentials and names both the tool and the interpreter."""
    for variable in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OLLAMA_API_KEY",
        "FURROW_ANTHROPIC_API_KEY",
        "FURROW_OPENAI_API_KEY",
    ):
        monkeypatch.delenv(variable, raising=False)

    result = runner.invoke(main, ["version"])

    assert result.exit_code == 0
    match = _VERSION_LINE.search(result.output)
    assert match is not None, result.output
    assert match["furrow"] == _package_version()
    assert match["python"] == platform.python_version()


# ----------------------------------------------------------------- usage errors


@pytest.mark.parametrize(
    "args",
    [
        ["start", "--model"],
        ["start", "goal", "--cycles", "abc"],
        ["start", "goal", "--provider", "bogus"],
        ["start", "goal", "--max-parallel", "0"],
        ["web", "--port", "99999"],
        ["start", "goal", "--test-timeout", "soon"],
        ["bogus-command"],
        ["version", "--no-such-option"],
    ],
)
def test_usage_errors_exit_2_instead_of_a_traceback(runner: CliRunner, args: list[str]) -> None:
    """Bad usage is reported by click as exit 2, never as an unhandled exception."""
    result = runner.invoke(main, args)

    assert result.exit_code == 2, result.output
    assert "Traceback" not in result.output
    assert isinstance(result.exception, SystemExit)


# ---------------------------------------------------------- settings, not globals


def test_build_settings_does_not_mutate_the_global_singleton(
    singleton: SingletonGuard, tmp_path: Path
) -> None:
    """``_build_settings`` returns a copy; the shared singleton is never written to."""
    settings = _build_settings(
        model="custom-model",
        provider="openai",
        workspace=tmp_path,
        max_cycles=7,
        max_parallel=3,
        test_command="pytest -q",
        test_timeout=12.5,
    )

    singleton.assert_unchanged()
    assert settings is not global_settings
    assert settings.model == "custom-model"
    assert settings.provider is Provider.OPENAI
    assert settings.workspace == tmp_path
    assert settings.max_cycles == 7
    assert settings.max_parallel_tasks == 3
    assert settings.test_command == "pytest -q"
    assert settings.test_timeout == 12.5


def test_start_overrides_reach_the_orchestrator_without_leaking_globally(
    runner: CliRunner, orchestrator: OrchestratorDouble, singleton: SingletonGuard, tmp_path: Path
) -> None:
    """A run uses the overrides it was given without changing the shared settings."""
    result = runner.invoke(
        main,
        [
            "start",
            "ship the console",
            "--model",
            "custom-model",
            "--provider",
            "openai",
            "--workspace",
            str(tmp_path),
            "--cycles",
            "4",
            "--max-parallel",
            "2",
            "--test-command",
            "pytest -q",
            "--test-timeout",
            "12.5",
        ],
    )

    assert result.exit_code == 0, result.output
    singleton.assert_unchanged()
    settings = orchestrator.only.settings
    assert settings.model == "custom-model"
    assert settings.provider is Provider.OPENAI
    assert settings.workspace == tmp_path
    assert settings.max_cycles == 4
    assert settings.max_parallel_tasks == 2
    assert settings.test_command == "pytest -q"
    assert settings.test_timeout == 12.5
    # The goal and the cycle budget also reached the run.
    assert orchestrator.only.goal == "ship the console"
    assert orchestrator.only.max_cycles == 4


def test_plan_overrides_reach_the_planner_without_leaking_globally(
    runner: CliRunner,
    planner: list[dict[str, Any]],
    orchestrator: OrchestratorDouble,
    singleton: SingletonGuard,
    tmp_path: Path,
) -> None:
    """``plan`` passes an overridden copy to the planner, not the singleton."""
    result = runner.invoke(
        main,
        ["plan", "ship the console", "--model", "custom-model", "--workspace", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    singleton.assert_unchanged()
    settings = planner[0]["settings"]
    assert settings is not global_settings
    assert settings.model == "custom-model"
    assert settings.workspace == tmp_path
    # Unpassed options keep the singleton's values rather than being blanked.
    assert settings.max_cycles == global_settings.max_cycles
    assert settings.provider == global_settings.provider
    assert orchestrator.built == []


def test_unset_options_leave_settings_fields_untouched(
    runner: CliRunner, orchestrator: OrchestratorDouble, singleton: SingletonGuard
) -> None:
    """With no options the built settings are the singleton, field for field."""
    result = runner.invoke(main, ["start", "plain goal"])

    assert result.exit_code == 0, result.output
    singleton.assert_unchanged()
    settings = orchestrator.only.settings
    assert settings.model_dump_json() == singleton.before
    for field in ("model", "provider", "max_cycles", "max_parallel_tasks", "test_timeout"):
        assert getattr(settings, field) == getattr(global_settings, field)


@pytest.mark.parametrize(
    ("option", "field", "value", "expected"),
    [
        ("--cycles", "max_cycles", "9", 9),
        ("--max-parallel", "max_parallel_tasks", "6", 6),
        ("--test-command", "test_command", "pytest -q", "pytest -q"),
        ("--test-timeout", "test_timeout", "30", 30.0),
    ],
)
def test_each_option_maps_onto_its_settings_field(
    runner: CliRunner,
    orchestrator: OrchestratorDouble,
    option: str,
    field: str,
    value: str,
    expected: Any,
) -> None:
    """Every override option lands on the settings field it advertises."""
    result = runner.invoke(main, ["start", "goal", option, value])

    assert result.exit_code == 0, result.output
    assert getattr(orchestrator.only.settings, field) == expected
    # Only the offered option changes; the other settings fields stay put.
    for other_option, other_field in SETTING_OPTIONS:
        if other_option == option:
            continue
        assert getattr(orchestrator.only.settings, other_field) == getattr(
            global_settings, other_field
        )


def test_workspace_option_maps_onto_the_workspace_field(
    runner: CliRunner, orchestrator: OrchestratorDouble, tmp_path: Path
) -> None:
    """``--workspace`` becomes a ``Path`` in the settings handed to the run."""
    result = runner.invoke(main, ["start", "goal", "--workspace", str(tmp_path)])

    assert result.exit_code == 0, result.output
    workspace = orchestrator.only.settings.workspace
    assert workspace == tmp_path
    assert isinstance(workspace, Path)


@pytest.mark.parametrize(
    ("typed", "expected"),
    [("openai", Provider.OPENAI), ("OpenAI", Provider.OPENAI), ("OLLAMA", Provider.OLLAMA)],
)
def test_provider_option_normalises_mixed_case(
    runner: CliRunner,
    orchestrator: OrchestratorDouble,
    typed: str,
    expected: Provider,
) -> None:
    """``--provider OpenAI`` is normalised to the enum value ``openai``."""
    result = runner.invoke(main, ["start", "goal", "--provider", typed])

    assert result.exit_code == 0, result.output
    assert orchestrator.only.settings.provider is expected


# ------------------------------------------------------------------- behaviour


def test_plan_prints_the_planned_tasks_and_executes_nothing(
    runner: CliRunner, planner: list[dict[str, Any]], orchestrator: OrchestratorDouble
) -> None:
    """``plan`` asks the planner and prints the task id and description as a table."""
    result = runner.invoke(main, ["plan", "ship the console"])

    assert result.exit_code == 0, result.output
    assert "task-1" in result.output
    assert "add the tests" in result.output
    assert planner[0]["goal"] == "ship the console"
    # Nothing has run yet, so no context is handed to the planner.
    assert planner[0]["context"] == ""
    assert orchestrator.built == []


def test_start_without_a_goal_prompts_for_one(
    runner: CliRunner, orchestrator: OrchestratorDouble, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing goal is collected with ``click.prompt`` and then used for the run."""
    prompted: list[str] = []
    real_prompt = click.prompt

    def spy(text: Any, *args: Any, **kwargs: Any) -> Any:
        prompted.append(text)
        return real_prompt(text, *args, **kwargs)

    monkeypatch.setattr(click, "prompt", spy)
    result = runner.invoke(main, ["start"], input="build a tiny cli\n")

    assert result.exit_code == 0, result.output
    assert prompted == ["Enter your goal for Furrow"]
    assert orchestrator.only.goal == "build a tiny cli"


def test_keyboard_interrupt_during_a_run_exits_the_documented_code(
    runner: CliRunner, orchestrator: OrchestratorDouble
) -> None:
    """A ``KeyboardInterrupt`` aborts with exit 130, no traceback, after asking to stop."""
    assert EXIT_INTERRUPTED == 130, "the documented interrupt exit code changed"
    orchestrator.raise_on_run = KeyboardInterrupt()

    result = runner.invoke(main, ["start", "goal"])

    assert result.exit_code == EXIT_INTERRUPTED
    assert "Traceback" not in result.output
    assert orchestrator.only.stop_requested is True


def test_a_stopped_run_exits_zero(runner: CliRunner, orchestrator: OrchestratorDouble) -> None:
    """Stopping on request is a clean exit 0 even though the goal was not met."""
    orchestrator.stop_during_run = True

    result = runner.invoke(main, ["start", "goal"])

    assert result.exit_code == 0, result.output
    assert "Traceback" not in result.output


def test_a_failing_run_exits_one_instead_of_raising(
    runner: CliRunner, orchestrator: OrchestratorDouble
) -> None:
    """An unexpected failure becomes a reported exit code, not a traceback."""
    orchestrator.raise_on_run = RuntimeError("boom")

    result = runner.invoke(main, ["start", "goal"])

    assert result.exit_code == 1, result.output
    assert "Traceback" not in result.output


def test_web_command_serves_the_console_with_the_requested_address(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``web --host/--port`` forwards them to the server runner verbatim."""
    served: dict[str, Any] = {}

    def fake_run(host: str = "0.0.0.0", port: int = 8000) -> None:
        served.update(host=host, port=port)

    monkeypatch.setattr("furrow.web.server.run", fake_run)
    result = runner.invoke(main, ["web", "--host", "127.0.0.1", "--port", "8123"])

    assert result.exit_code == 0, result.output
    assert served == {"host": "127.0.0.1", "port": 8123}