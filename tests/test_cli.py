"""Tests for the command line surface in :mod:`furrow.cli.main`.

HEADLINE REGRESSION (bug #13): ``--model`` used to set only ``Settings.model``, a field no
agent ever reads -- every agent uses ``planner_model`` / ``worker_model`` / ``tester_model``.
The flag was therefore a complete silent no-op: the CLI accepted it, changed a setting
nobody consumed, and the run used the configured per-role defaults anyway.
:func:`test_model_flag_reaches_every_agent_model` is the load-bearing assertion here.

Everything is driven through ``click.testing.CliRunner`` and
``furrow.cli.main.Orchestrator`` is replaced by :class:`FakeOrchestrator`, which records the
``settings``/``bus``/``goal`` it was constructed with. No LLM, no network, no subprocess.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from furrow.cli import main as cli_main
from furrow.config import Provider
from furrow.core.orchestrator import RunSummary
from furrow.events import EventBus, EventType
from furrow.llm import LLMConfigurationError, LLMError
from furrow.web import server as web_server
from tests.conftest import ScriptedLLMClient, as_llm_client

GOAL = "ship the thing"


# ---------------------------------------------------------------------------
# doubles
# ---------------------------------------------------------------------------


class FakeOrchestrator:
    """Records what it was built with and returns immediately.

    Mirrors the real ``Orchestrator`` surface the CLI touches: ``run()``, ``stop()`` and
    ``snapshot()`` (whose keys ``_print_summary`` reads), plus ``bus`` so the live feed and
    the subscriber-leak assertion have something real to look at.
    """

    PRE_EVENTS = (
        (EventType.RUN_START, "Goal: {goal}"),
        (EventType.CYCLE_START, "Cycle 1"),
        (EventType.PLAN, "Planned 1 task(s)"),
        (EventType.TASK_START, "Task 1: do the thing"),
        (EventType.TASK_END, "Task 1 completed"),
    )
    POST_EVENTS = (
        (EventType.TEST_RESULT, "1 passed"),
        (EventType.RUN_END, "Furrow halted: goal_complete"),
    )

    def __init__(
        self,
        goal: str,
        client: Any = None,
        settings: Any = None,
        bus: EventBus | None = None,
    ) -> None:
        self.goal = goal
        self.client = client
        self.settings = settings
        self.bus = bus if bus is not None else EventBus(500)
        self.stop_calls = 0
        self.error: BaseException | None = None
        self.hold_until_stopped = False
        self.completed = False
        self.cycles = 1
        self._stop_requested = asyncio.Event()
        self._halted_reason: str | None = None

    @property
    def halted_reason(self) -> str | None:
        return self._halted_reason

    async def run(self) -> RunSummary:
        try:
            self._emit_all(self.PRE_EVENTS)
            if self.hold_until_stopped:
                # Only released by stop(): a Ctrl-C test must have a run to stop.
                await self._stop_requested.wait()
            self._emit_all(self.POST_EVENTS)
            if self.error is not None:
                raise self.error
            self._halted_reason = "goal_complete"
            return RunSummary(
                cycles=self.cycles,
                tasks_total=1,
                tasks_completed=1,
                tasks_failed=0,
                tests_passed=True,
                halted_reason="goal_complete",
            )
        finally:
            self.completed = True

    async def stop(self) -> None:
        self.stop_calls += 1
        self._halted_reason = "stopped"
        self._stop_requested.set()

    def snapshot(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "cycles": self.cycles,
            "halted_reason": "goal_complete",
            "running": not self.completed,
            "tasks": [{"id": "1", "status": "completed"}],
            "test_results": [{"passed": True, "summary": "1 passed", "failures": []}],
            "usage": {"input_tokens": 5, "output_tokens": 2, "requests": 1},
        }

    def _emit_all(self, pairs: tuple[tuple[EventType, str], ...]) -> None:
        for event_type, template in pairs:
            self.bus.emit(
                event_type,
                template.format(goal=self.goal),
                cycle=self.cycles,
                id="1",
                description="do the thing",
                status="completed",
                passed=True,
                summary="1 passed",
                failures=[],
                halted_reason="goal_complete",
                new_tasks=1,
                tracked_tasks=1,
            )


def install_fake(
    monkeypatch: pytest.MonkeyPatch,
    *,
    error: BaseException | None = None,
    hold_until_stopped: bool = False,
) -> list[FakeOrchestrator]:
    """Swap ``furrow.cli.main.Orchestrator`` for :class:`FakeOrchestrator`."""
    created: list[FakeOrchestrator] = []

    class Installed(FakeOrchestrator):
        def __init__(
            self,
            goal: str,
            client: Any = None,
            settings: Any = None,
            bus: EventBus | None = None,
        ) -> None:
            super().__init__(goal, client=client, settings=settings, bus=bus)
            self.error = error
            self.hold_until_stopped = hold_until_stopped
            created.append(self)

    monkeypatch.setattr(cli_main, "Orchestrator", Installed)
    return created


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def start_args(workspace: Path, *extra: str) -> list[str]:
    """`start` with the goal plus extra flags, always scoped to the tmp workspace."""
    return ["start", GOAL, "--workspace", str(workspace), *extra]


def only_settings(created: list[FakeOrchestrator]) -> Any:
    """The ``Settings`` the single constructed orchestrator was handed."""
    assert len(created) == 1, created
    return created[0].settings


def clean_failure(result: Any) -> None:
    """A CLI failure must be a message, never a traceback."""
    assert "Traceback" not in result.output, result.output
    assert "most recent call last" not in result.output, result.output


def flat(result: Any) -> str:
    """`result.output` with rich's soft line wraps removed, so long paths stay contiguous."""
    return result.output.replace("\n", "")


# ---------------------------------------------------------------------------
# regression #13: --model used to be a silent no-op
# ---------------------------------------------------------------------------


def test_model_flag_reaches_every_agent_model(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, runner: CliRunner
) -> None:
    """REGRESSION: ``--model`` must set the four fields agents actually read.

    ``--model`` used to write only ``Settings.model``, which no agent consults, so the flag
    looked accepted and changed nothing.
    """
    created = install_fake(monkeypatch)
    result = runner.invoke(cli_main.main, start_args(workspace, "--model", "shared-model"))
    assert result.exit_code == 0, result.output
    settings = only_settings(created)
    assert settings.model == "shared-model"
    assert settings.planner_model == "shared-model"
    assert settings.worker_model == "shared-model"
    assert settings.tester_model == "shared-model"


def test_per_role_model_flags_win_over_the_shared_model(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, runner: CliRunner
) -> None:
    """``--planner-model Y`` overrides only the planner; the rest keep ``--model X``."""
    created = install_fake(monkeypatch)
    result = runner.invoke(
        cli_main.main,
        start_args(
            workspace,
            "--model",
            "shared-model",
            "--planner-model",
            "planner-only",
        ),
    )
    assert result.exit_code == 0, result.output
    settings = only_settings(created)
    assert settings.planner_model == "planner-only"
    assert settings.model == "shared-model"
    assert settings.worker_model == "shared-model"
    assert settings.tester_model == "shared-model"


def test_each_role_flag_alone_reaches_only_its_own_field(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, runner: CliRunner
) -> None:
    """Every role flag is independently honoured and does not drag the others with it."""
    created = install_fake(monkeypatch)
    result = runner.invoke(
        cli_main.main,
        start_args(
            workspace,
            "--worker-model",
            "worker-only",
            "--tester-model",
            "tester-only",
        ),
    )
    assert result.exit_code == 0, result.output
    settings = only_settings(created)
    assert settings.worker_model == "worker-only"
    assert settings.tester_model == "tester-only"
    assert settings.model != "worker-only"
    assert settings.planner_model not in {"worker-only", "tester-only"}


def test_model_flag_does_not_leak_into_the_next_run(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, runner: CliRunner
) -> None:
    """Overrides are per invocation: a later run gets the configured defaults back."""
    created = install_fake(monkeypatch)
    first = runner.invoke(cli_main.main, start_args(workspace, "--model", "shared-model"))
    assert first.exit_code == 0, first.output
    second = runner.invoke(cli_main.main, start_args(workspace))
    assert second.exit_code == 0, second.output
    assert len(created) == 2
    assert created[0].settings.planner_model == "shared-model"
    assert created[1].settings.planner_model != "shared-model"
    assert created[1].settings.model != "shared-model"


# ---------------------------------------------------------------------------
# remaining start flags
# ---------------------------------------------------------------------------


def test_provider_reaches_settings(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, runner: CliRunner
) -> None:
    created = install_fake(monkeypatch)
    result = runner.invoke(cli_main.main, start_args(workspace, "--provider", "openai"))
    assert result.exit_code == 0, result.output
    assert only_settings(created).provider == Provider.OPENAI


def test_invalid_provider_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, runner: CliRunner
) -> None:
    created = install_fake(monkeypatch)
    result = runner.invoke(cli_main.main, start_args(workspace, "--provider", "nope"))
    assert result.exit_code != 0
    clean_failure(result)
    assert "nope" in result.output
    assert "--provider" in result.output
    assert "anthropic" in result.output and "ollama" in result.output
    assert created == [], "a rejected provider must not start a run"


def test_workspace_reaches_settings(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, runner: CliRunner
) -> None:
    created = install_fake(monkeypatch)
    result = runner.invoke(cli_main.main, start_args(workspace))
    assert result.exit_code == 0, result.output
    assert only_settings(created).workspace == Path(workspace)


def test_missing_workspace_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, runner: CliRunner
) -> None:
    created = install_fake(monkeypatch)
    missing = workspace / "definitely" / "not" / "here"
    result = runner.invoke(
        cli_main.main, ["start", GOAL, "--workspace", str(missing)]
    )
    assert result.exit_code != 0
    clean_failure(result)
    assert "does not exist or is not a directory" in flat(result)
    assert str(missing) in flat(result)
    assert created == [], "a rejected workspace must not start a run"


def test_a_file_is_not_a_valid_workspace(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, runner: CliRunner
) -> None:
    """The check is "exists AND is a directory", not just "exists"."""
    created = install_fake(monkeypatch)
    target = workspace / "a-file.txt"
    target.write_text("not a directory", encoding="utf-8")
    result = runner.invoke(cli_main.main, ["start", GOAL, "--workspace", str(target)])
    assert result.exit_code != 0
    clean_failure(result)
    assert "not a directory" in flat(result)
    assert created == []


def test_numeric_and_path_flags_reach_settings(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, runner: CliRunner
) -> None:
    """``--max-cycles`` / ``--max-parallel`` / ``--max-tool-iterations`` / ``--log-level`` /
    ``--state-file`` each land on the settings object the orchestrator receives."""
    created = install_fake(monkeypatch)
    state_file = workspace / "state" / "run.json"
    result = runner.invoke(
        cli_main.main,
        start_args(
            workspace,
            "--max-cycles",
            "7",
            "--max-parallel",
            "3",
            "--max-tool-iterations",
            "4",
            "--log-level",
            "DEBUG",
            "--state-file",
            str(state_file),
        ),
    )
    assert result.exit_code == 0, result.output
    settings = only_settings(created)
    assert settings.max_cycles == 7
    assert settings.max_parallel_tasks == 3
    assert settings.max_tool_iterations == 4
    assert settings.log_level == "DEBUG"
    assert settings.state_file == state_file
    # The fake never persists anything, so nothing was written.
    assert not state_file.exists()


def test_unlimited_max_cycles_is_accepted(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, runner: CliRunner
) -> None:
    """``--max-cycles 0`` means "unlimited" on the CLI (unlike the web API, which 400s)."""
    created = install_fake(monkeypatch)
    result = runner.invoke(cli_main.main, start_args(workspace, "--max-cycles", "0"))
    assert result.exit_code == 0, result.output
    assert only_settings(created).max_cycles == 0


def test_the_goal_is_passed_to_the_orchestrator(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, runner: CliRunner
) -> None:
    created = install_fake(monkeypatch)
    result = runner.invoke(
        cli_main.main, ["start", "an exact goal string", "--workspace", str(workspace)]
    )
    assert result.exit_code == 0, result.output
    assert only_settings(created) is created[0].settings
    assert created[0].goal == "an exact goal string"


def test_a_missing_goal_is_prompted_for(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, runner: CliRunner
) -> None:
    created = install_fake(monkeypatch)
    result = runner.invoke(
        cli_main.main, ["start", "--workspace", str(workspace)], input="typed goal\n"
    )
    assert result.exit_code == 0, result.output
    assert created[0].goal == "typed goal"


# ---------------------------------------------------------------------------
# interrupt and error handling
# ---------------------------------------------------------------------------


def test_first_keyboard_interrupt_stops_the_run_and_still_prints_the_summary(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, runner: CliRunner
) -> None:
    """A first Ctrl-C asks the run to stop, the run finishes and the summary still prints.

    The KeyboardInterrupt is injected at ``_render_event``, the point where a real Ctrl-C
    lands in the main thread while the live feed is being drawn. That is the only way to get
    an *in-coroutine* KeyboardInterrupt on CPython 3.10: raising one inside a task escapes
    ``asyncio.run`` outright (``Task.__step`` re-raises KeyboardInterrupt into the loop), so
    ``_start_impl``'s own handler would never run. The double deliberately does not raise
    the interrupt itself -- it stays running until ``stop()`` releases it, which is exactly
    the graceful-shutdown path under test.
    """
    created = install_fake(monkeypatch, hold_until_stopped=True)
    original = cli_main._render_event
    state = {"raised": False}

    def interrupting_render(event: Any) -> None:
        if not state["raised"]:
            state["raised"] = True
            raise KeyboardInterrupt
        original(event)

    monkeypatch.setattr(cli_main, "_render_event", interrupting_render)

    result = runner.invoke(cli_main.main, start_args(workspace))

    assert result.exit_code == 0, result.output
    clean_failure(result)
    assert state["raised"] is True
    assert len(created) == 1
    assert created[0].stop_calls == 1, "the first interrupt must request a graceful stop"
    assert created[0].completed is True, "the run must finish after stop()"
    assert "Run Summary" in result.output
    assert "Cycles: 1" in result.output
    assert "Halted reason: goal_complete" in result.output
    assert len(created[0].bus._subscribers) == 0


def test_llm_configuration_error_is_a_clean_non_zero_exit(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, runner: CliRunner
) -> None:
    created = install_fake(
        monkeypatch, error=LLMConfigurationError("no Anthropic API key configured")
    )
    result = runner.invoke(cli_main.main, start_args(workspace))
    assert result.exit_code == 1
    clean_failure(result)
    assert "LLM configuration error: no Anthropic API key configured" in result.output
    assert len(result.output.strip().splitlines()) >= 1
    assert created[0].completed is True
    assert len(created[0].bus._subscribers) == 0


def test_llm_error_is_a_clean_non_zero_exit(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, runner: CliRunner
) -> None:
    created = install_fake(monkeypatch, error=LLMError("provider returned 500"))
    result = runner.invoke(cli_main.main, start_args(workspace))
    assert result.exit_code == 1
    clean_failure(result)
    assert "LLM error: provider returned 500" in result.output
    assert len(created) == 1
    assert len(created[0].bus._subscribers) == 0


def test_an_unexpected_error_keeps_its_traceback(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, runner: CliRunner
) -> None:
    """An unknown exception is a bug, so it is reported with a traceback and exit code 1."""
    created = install_fake(monkeypatch, error=ZeroDivisionError("division by zero"))
    result = runner.invoke(cli_main.main, start_args(workspace))
    assert result.exit_code == 1
    assert "ZeroDivisionError" in result.output
    assert len(created) == 1


# ---------------------------------------------------------------------------
# live feed
# ---------------------------------------------------------------------------


def test_live_feed_renders_cycle_task_and_test_lines(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, runner: CliRunner
) -> None:
    """The default (live) feed draws cycle, task and test lines and releases its subscriber."""
    created = install_fake(monkeypatch)
    result = runner.invoke(cli_main.main, start_args(workspace))
    assert result.exit_code == 0, result.output
    output = result.output
    assert "=== Cycle 1 ===" in output
    assert "Task 1 started" in output
    assert "Task 1 completed" in output
    assert "Tests passed: 1 passed" in output
    assert "Furrow halted: goal_complete" in output
    assert "Run Summary" in output
    assert "Cycles: 1" in output
    assert "Tasks: 1 total, 1 completed, 0 failed" in output
    assert "Tests passed: True" in output
    assert "Tokens: 5 in, 2 out, 1 requests" in output
    assert len(created[0].bus._subscribers) == 0, "the live feed leaked a bus subscriber"


def test_no_live_output_suppresses_the_feed_but_keeps_the_summary(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, runner: CliRunner
) -> None:
    created = install_fake(monkeypatch)
    result = runner.invoke(cli_main.main, start_args(workspace, "--no-live-output"))
    assert result.exit_code == 0, result.output
    assert "=== Cycle 1 ===" not in result.output
    assert "Task 1 started" not in result.output
    assert "Run Summary" in result.output
    assert len(created[0].bus._subscribers) == 0


def test_a_failing_test_run_is_rendered_in_red_text(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, runner: CliRunner
) -> None:
    """The feed distinguishes a failing verdict, lists the failures and is not a crash."""

    class Failing(FakeOrchestrator):
        async def run(self) -> RunSummary:
            self.bus.emit(
                EventType.TEST_RESULT,
                "2 failed",
                cycle=1,
                passed=False,
                summary="2 failed",
                failures=["tests/test_a.py::test_one", "tests/test_a.py::test_two"],
            )
            self.bus.emit(EventType.ERROR, "Tester failed: boom", cycle=1, error="boom")
            self.bus.emit(
                EventType.RUN_END,
                "Furrow halted: max_cycles",
                cycle=1,
                halted_reason="max_cycles",
            )
            self.completed = True
            self._halted_reason = "max_cycles"
            return RunSummary(1, 1, 0, 1, False, "max_cycles")

        def snapshot(self) -> dict[str, Any]:
            base = super().snapshot()
            base["test_results"] = [
                {
                    "passed": False,
                    "summary": "2 failed",
                    "failures": ["tests/test_a.py::test_one"],
                }
            ]
            base["tasks"] = [{"id": "1", "status": "failed"}]
            return base

    monkeypatch.setattr(cli_main, "Orchestrator", Failing)
    result = runner.invoke(cli_main.main, start_args(workspace))
    assert result.exit_code == 0, result.output
    assert "Tests failed: 2 failed" in result.output
    assert "tests/test_a.py::test_one" in result.output
    assert "Tester failed: boom" in result.output
    assert "Furrow halted: max_cycles" in result.output
    assert "Tasks: 1 total, 0 completed, 1 failed" in result.output
    assert "Tests passed: False" in result.output


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def test_plan_renders_the_task_table_without_building_agents(
    monkeypatch: pytest.MonkeyPatch, fake_client: ScriptedLLMClient, runner: CliRunner
) -> None:
    """`plan` prints a plan table and nothing else: no orchestrator, worker or tester."""
    created = install_fake(monkeypatch)
    monkeypatch.setattr(
        cli_main, "LLMClient", lambda settings=None: as_llm_client(fake_client)
    )

    class _NeverBuilt:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("the plan command must not build a worker or a tester")

    monkeypatch.setattr("furrow.agents.worker.WorkerAgent", _NeverBuilt)
    monkeypatch.setattr("furrow.agents.tester.TesterAgent", _NeverBuilt)

    result = runner.invoke(cli_main.main, ["plan", GOAL])

    assert result.exit_code == 0, result.output
    clean_failure(result)
    output = result.output
    assert "Plan:" in output
    assert GOAL in output
    for header in ("ID", "Description", "Files", "Dependencies"):
        assert header in output, output
    assert "Add a greeting module" in output, output
    assert "greeting.py" in output, output
    assert "Rationale" in output
    assert "One small, self-contained task." in output
    assert created == [], "plan must not construct an orchestrator"
    # The planner really was called, once, through the scripted client.
    assert [call["method"] for call in fake_client.calls] == ["complete_json"]


def test_plan_reports_an_llm_failure_cleanly(
    monkeypatch: pytest.MonkeyPatch, make_client: Any, runner: CliRunner
) -> None:
    client = make_client(error=LLMConfigurationError("no key for provider"))
    monkeypatch.setattr(cli_main, "LLMClient", lambda settings=None: as_llm_client(client))
    result = runner.invoke(cli_main.main, ["plan", GOAL])
    assert result.exit_code == 1
    clean_failure(result)
    assert "LLM configuration error: no key for provider" in result.output


# ---------------------------------------------------------------------------
# web
# ---------------------------------------------------------------------------


def test_web_delegates_to_the_server_run(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner
) -> None:
    """`furrow web` must reach ``furrow.web.server.run`` (monkeypatched: never called here)."""
    calls: list[tuple[str | None, int | None]] = []
    monkeypatch.setattr(
        web_server,
        "run",
        lambda host=None, port=None, **kwargs: calls.append((host, port)),
    )
    result = runner.invoke(cli_main.main, ["web", "--host", "0.0.0.0", "--port", "9123"])
    assert result.exit_code == 0, result.output
    assert calls == [("0.0.0.0", 9123)]


def test_web_defaults_come_from_settings(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner
) -> None:
    """No ``--host``/``--port`` means ``Settings.web_host`` / ``Settings.web_port``."""
    calls: list[tuple[str | None, int | None]] = []
    monkeypatch.setattr(
        web_server,
        "run",
        lambda host=None, port=None, **kwargs: calls.append((host, port)),
    )
    result = runner.invoke(cli_main.main, ["web"])
    assert result.exit_code == 0, result.output
    from furrow.config import get_settings

    settings = get_settings()
    assert calls == [(settings.web_host, settings.web_port)]
    assert calls[0][0] == "127.0.0.1", "the privileged surface must not default to 0.0.0.0"


def test_web_port_alone_keeps_the_configured_host(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner
) -> None:
    calls: list[tuple[str | None, int | None]] = []
    monkeypatch.setattr(
        web_server,
        "run",
        lambda host=None, port=None, **kwargs: calls.append((host, port)),
    )
    result = runner.invoke(cli_main.main, ["web", "--port", "9999"])
    assert result.exit_code == 0, result.output
    host, port = calls[0]
    assert port == 9999
    assert host == "127.0.0.1"


def test_web_never_starts_a_run_itself(monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    """`web` must not touch the orchestrator: the server does that per request."""
    created = install_fake(monkeypatch)
    monkeypatch.setattr(web_server, "run", lambda host=None, port=None, **kwargs: None)
    result = runner.invoke(cli_main.main, ["web"])
    assert result.exit_code == 0, result.output
    assert created == []


# ---------------------------------------------------------------------------
# version and help
# ---------------------------------------------------------------------------


def test_version_prints_the_package_version(runner: CliRunner) -> None:
    result = runner.invoke(cli_main.main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output
    assert "furrow" in result.output


def test_group_help_lists_every_command(runner: CliRunner) -> None:
    result = runner.invoke(cli_main.main, ["--help"])
    assert result.exit_code == 0
    assert "start" in result.output
    assert "plan" in result.output
    assert "web" in result.output
    assert "Furrow" in result.output


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("start", ("--model", "--planner-model", "--worker-model", "--tester-model")),
        ("web", ("--host", "--port", "--reload")),
        ("plan", ()),
    ],
)
def test_subcommand_help_exits_zero(
    runner: CliRunner, command: str, expected: tuple[str, ...]
) -> None:
    result = runner.invoke(cli_main.main, [command, "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    for flag in expected:
        assert flag in result.output, result.output


def test_help_does_not_start_a_run(monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    created = install_fake(monkeypatch)
    assert runner.invoke(cli_main.main, ["start", "--help"]).exit_code == 0
    assert runner.invoke(cli_main.main, ["plan", "--help"]).exit_code == 0
    assert created == []


def test_no_arguments_prints_help_without_starting_a_run(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner
) -> None:
    """The group never runs a goal on its own."""
    created = install_fake(monkeypatch)
    result = runner.invoke(cli_main.main, [])
    clean_failure(result)
    assert "Usage:" in result.output
    assert created == []


def test_main_module_exposes_the_group() -> None:
    """``furrow = furrow.cli.main:main`` must resolve to a click group."""
    assert isinstance(cli_main.main, __import__("click").Group)
    assert json.dumps(sorted(cli_main.main.commands)) == json.dumps(["plan", "start", "web"])
