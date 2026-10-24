"""Tests for the Furrow command line interface.

Nothing here may touch the network or start a real run: the orchestrator is
always replaced by :class:`FakeOrchestrator`.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import pytest
from click.testing import CliRunner

from furrow.cli.main import main
from furrow.config import RunResult, TaskModel, Usage
from furrow.core.state import RunState, StateStore


class FakeOrchestrator:
    """Stand-in for the real orchestrator: records its kwargs, never runs."""

    init_kwargs: ClassVar[dict[str, Any]] = {}
    error: ClassVar[BaseException | None] = None
    result: ClassVar[RunResult | None] = None

    def __init__(
        self,
        goal: str,
        client: Any = None,
        settings: Any = None,
        state_store: Any = None,
        event_bus: Any = None,
        resume: bool = False,
    ) -> None:
        FakeOrchestrator.init_kwargs = {
            "goal": goal,
            "client": client,
            "settings": settings,
            "state_store": state_store,
            "event_bus": event_bus,
            "resume": resume,
        }
        self.goal = goal

    async def run(self) -> RunResult:
        if FakeOrchestrator.error is not None:
            raise FakeOrchestrator.error
        if FakeOrchestrator.result is not None:
            return FakeOrchestrator.result
        return RunResult(
            passed=True,
            cycles=1,
            tasks_completed=1,
            summary="done",
            stopped_reason="goal complete",
        )


@pytest.fixture
def fake_orchestrator(monkeypatch: pytest.MonkeyPatch) -> type[FakeOrchestrator]:
    """Replace the orchestrator in the CLI with the fake one."""
    FakeOrchestrator.init_kwargs = {}
    FakeOrchestrator.error = None
    FakeOrchestrator.result = None
    monkeypatch.setattr("furrow.cli.main.Orchestrator", FakeOrchestrator)
    return FakeOrchestrator


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _write_state(workspace: Path) -> RunState:
    """Persist a run state with a single task inside *workspace*."""
    state = RunState(
        goal="ship it",
        original_goal="ship it",
        cycles=2,
        tasks=[TaskModel(id="t1", description="do the thing", status="completed")],
    )
    StateStore(workspace / ".furrow" / "state.json").save(state)
    return state


def _make_result(**kwargs: Any) -> RunResult:
    """Build a :class:`RunResult` for the summary tests.

    The newer fields (``tasks_blocked``, ``tests_run``) are set on the
    instance instead of being handed to the constructor: they may not be
    declared on the model yet and the CLI reads them defensively anyway.
    """
    result = RunResult(
        passed=kwargs.pop("passed", False),
        cycles=kwargs.pop("cycles", 1),
        tasks_completed=kwargs.pop("tasks_completed", 0),
        tasks_failed=kwargs.pop("tasks_failed", 0),
        summary=kwargs.pop("summary", ""),
        stopped_reason=kwargs.pop("stopped_reason", ""),
        skipped=kwargs.pop("skipped", False),
    )
    for name, value in kwargs.items():
        object.__setattr__(result, name, value)
    return result


def test_start_requires_goal_prompt(
    runner: CliRunner, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    result = runner.invoke(main, ["start"], input="")

    assert "Enter your goal for Furrow" in result.output
    assert fake_orchestrator.init_kwargs == {}


def test_start_prompts_and_uses_the_typed_goal(
    runner: CliRunner, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    result = runner.invoke(main, ["start"], input="build a thing\n")

    assert result.exit_code == 0
    assert "Enter your goal for Furrow" in result.output
    assert fake_orchestrator.init_kwargs["goal"] == "build a thing"


def test_status_without_state(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(main, ["status", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "No furrow run state found" in result.output


def test_status_with_state(runner: CliRunner, tmp_path: Path) -> None:
    _write_state(tmp_path)

    result = runner.invoke(main, ["status", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "ship it" in result.output
    assert "t1" in result.output
    assert "completed" in result.output
    assert "do the thing" in result.output


def test_resume_without_state(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(main, ["resume", "--workspace", str(tmp_path)])

    assert result.exit_code == 1
    assert "No furrow run state found to resume." in result.output


def test_resume_runs_orchestrator(
    runner: CliRunner, fake_orchestrator: type[FakeOrchestrator], tmp_path: Path
) -> None:
    _write_state(tmp_path)

    result = runner.invoke(main, ["resume", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert fake_orchestrator.init_kwargs["resume"] is True
    assert fake_orchestrator.init_kwargs["goal"] == "ship it"
    assert "PASSED" in result.output
    assert "goal complete" in result.output


def test_start_max_cycles_negative_rejected(
    runner: CliRunner, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    result = runner.invoke(main, ["start", "build a thing", "--max-cycles", "-1"])

    assert result.exit_code != 0


def test_resume_max_cycles_negative_rejected(
    runner: CliRunner, fake_orchestrator: type[FakeOrchestrator], tmp_path: Path
) -> None:
    _write_state(tmp_path)

    result = runner.invoke(
        main, ["resume", "--workspace", str(tmp_path), "--max-cycles", "-1"]
    )

    assert result.exit_code != 0


def test_web_reports_friendly_error_on_bind_failure(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_error(*args: Any, **kwargs: Any) -> None:
        raise OSError("address already in use")

    monkeypatch.setattr("furrow.web.server.run", raise_error)

    result = runner.invoke(main, ["web"])

    assert result.exit_code == 1
    assert "Furrow failed:" in result.output


def test_start_exits_1_on_failed_run(
    runner: CliRunner, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    fake_orchestrator.result = _make_result(passed=False, tests_run=5)

    result = runner.invoke(main, ["start", "build a thing"])

    assert result.exit_code == 1
    assert "FAILED" in result.output


def test_start_exits_0_on_passed_run(
    runner: CliRunner, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    fake_orchestrator.result = _make_result(passed=True, tests_run=3)

    result = runner.invoke(main, ["start", "build a thing"])

    assert result.exit_code == 0
    assert "PASSED" in result.output


def test_start_exits_0_on_skipped_run(
    runner: CliRunner, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    fake_orchestrator.result = _make_result(passed=False, skipped=True, tests_run=0)

    result = runner.invoke(main, ["start", "build a thing"])

    assert result.exit_code == 0
    assert "SKIPPED" in result.output


def test_start_uses_max_cycles_and_workspace(
    runner: CliRunner, fake_orchestrator: type[FakeOrchestrator], tmp_path: Path
) -> None:
    from furrow.config import settings as global_settings

    max_cycles_before = global_settings.max_cycles
    workspace_before = global_settings.workspace

    result = runner.invoke(
        main,
        ["start", "build a thing", "--max-cycles", "4", "--workspace", str(tmp_path)],
    )

    assert result.exit_code == 0
    kwargs = fake_orchestrator.init_kwargs
    assert kwargs["goal"] == "build a thing"
    assert kwargs["resume"] is False
    run_settings = kwargs["settings"]
    assert run_settings.max_cycles == 4
    assert run_settings.workspace == Path(tmp_path)
    assert kwargs["state_store"].path == tmp_path / ".furrow" / "state.json"
    # The global settings must not be mutated by a single run.
    assert global_settings.max_cycles == max_cycles_before
    assert global_settings.workspace == workspace_before


def test_start_stops_gracefully_on_keyboard_interrupt(
    runner: CliRunner, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    fake_orchestrator.error = KeyboardInterrupt()

    result = runner.invoke(main, ["start", "build a thing"])

    assert result.exit_code == 0
    assert "Furrow stopped by user." in result.output


def test_start_reports_failures(
    runner: CliRunner, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    fake_orchestrator.error = RuntimeError("boom")

    result = runner.invoke(main, ["start", "build a thing"])

    assert result.exit_code == 1
    assert "boom" in result.output


def test_main_passes_configured_log_level_to_configure_logging(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FURROW_LOG_LEVEL (read from settings at call time) reaches configure_logging.

    main() used to call configure_logging() with no level, silently dropping
    settings.log_level.
    """
    monkeypatch.setattr("furrow.cli.main.settings.log_level", "DEBUG")

    captured: list[str] = []

    def fake_configure_logging(level: str) -> None:
        captured.append(level)

    monkeypatch.setattr("furrow.cli.main.configure_logging", fake_configure_logging)

    result = runner.invoke(main, ["status", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert captured == ["DEBUG"]


def test_summary_prints_skipped_when_no_test_runner(
    runner: CliRunner, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    """A skipped run prints a distinct SKIPPED line instead of PASSED."""
    FakeOrchestrator.result = RunResult(
        passed=True,
        cycles=1,
        summary="stopped: goal complete",
        skipped=True,
    )

    output = runner.invoke(main, ["start", "build a thing"]).output

    assert "SKIPPED" in output
    assert "PASSED" not in output


def test_summary_prints_passed_by_default(
    runner: CliRunner, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    """Without the skipped flag the usual PASSED line is printed."""
    result = RunResult(
        passed=True,
        cycles=1,
        tasks_completed=1,
        summary="done",
        stopped_reason="goal complete",
    )
    assert result.skipped is False
    FakeOrchestrator.result = result

    output = runner.invoke(main, ["start", "build a thing"]).output

    assert "PASSED" in output
    assert "SKIPPED" not in output


def test_summary_prints_usage_when_result_carries_usage(
    runner: CliRunner, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    """The summary prints the LLM usage line when the result carries usage."""
    FakeOrchestrator.result = RunResult(
        passed=True,
        cycles=1,
        tasks_completed=1,
        summary="done",
        stopped_reason="goal complete",
        usage=Usage(input_tokens=3900, output_tokens=421, calls=12),
    )

    output = runner.invoke(main, ["start", "build a thing"]).output

    assert "LLM usage:" in output
    assert "12 calls" in output
    assert "4321 tokens" in output
    assert "in 3900" in output
    assert "out 421" in output


def test_summary_prints_cost_when_carried(
    runner: CliRunner, fake_orchestrator: type[FakeOrchestrator]
):
    """The summary prints the estimated cost when the result carries a non-zero cost."""
    FakeOrchestrator.result = RunResult(
        passed=True,
        cycles=1,
        tasks_completed=1,
        summary="done",
        stopped_reason="goal complete",
        usage=Usage(input_tokens=3900, output_tokens=421, calls=12, cost_usd=0.123456),
    )

    output = runner.invoke(main, ["start", "build a thing"]).output

    assert "LLM usage:" in output
    assert "Estimated cost:" in output
    assert "$0.123456" in output


def test_summary_omits_cost_when_zero(
    runner: CliRunner, fake_orchestrator: type[FakeOrchestrator]
):
    """A run with zero estimated cost does not print a cost line."""
    FakeOrchestrator.result = RunResult(
        passed=True,
        cycles=1,
        tasks_completed=1,
        summary="done",
        stopped_reason="goal complete",
        usage=Usage(input_tokens=3900, output_tokens=421, calls=12, cost_usd=0.0),
    )

    output = runner.invoke(main, ["start", "build a thing"]).output

    assert "LLM usage:" in output
    assert "Estimated cost:" not in output


def test_status_prints_persisted_cost(
    runner: CliRunner, tmp_path: Path
):
    """The status command prints the estimated cost persisted in the run state."""
    state = RunState(
        goal="ship it",
        original_goal="ship it",
        cycles=2,
        usage=Usage(input_tokens=3900, output_tokens=421, calls=12, cost_usd=0.25),
        tasks=[TaskModel(id="t1", description="do the thing", status="completed")],
    )
    StateStore(tmp_path / ".furrow" / "state.json").save(state)

    result = runner.invoke(main, ["status", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "LLM usage:" in result.output
    assert "Estimated cost:" in result.output
    assert "$0.250000" in result.output


def test_summary_skips_usage_line_when_zero_calls(
    runner: CliRunner, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    """A run with no LLM calls does not print a misleading zero usage line."""
    FakeOrchestrator.result = RunResult(
        passed=True,
        cycles=1,
        tasks_completed=1,
        summary="done",
        stopped_reason="goal complete",
        usage=Usage(),
    )
    assert FakeOrchestrator.result.usage.calls == 0

    output = runner.invoke(main, ["start", "build a thing"]).output

    assert "LLM usage" not in output


def test_summary_prints_files_changed_when_present(
    runner: CliRunner, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    FakeOrchestrator.result = RunResult(
        passed=True,
        cycles=1,
        tasks_completed=1,
        summary="done",
        stopped_reason="goal complete",
        files_changed=["furrow/core/orchestrator.py", "furrow/config.py"],
    )

    output = runner.invoke(main, ["start", "build a thing"]).output

    assert "Files changed:" in output
    assert "furrow/core/orchestrator.py, furrow/config.py" in output


def test_summary_omits_files_changed_when_empty(
    runner: CliRunner, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    FakeOrchestrator.result = RunResult(
        passed=True,
        cycles=1,
        tasks_completed=1,
        summary="done",
        stopped_reason="goal complete",
        files_changed=[],
    )

    output = runner.invoke(main, ["start", "build a thing"]).output

    assert "Files changed:" not in output


def test_summary_prints_tasks_blocked(
    runner: CliRunner, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    """The summary prints the number of blocked tasks."""
    FakeOrchestrator.result = _make_result(passed=True, tasks_blocked=3)

    output = runner.invoke(main, ["start", "build a thing"]).output

    assert re.search(r"Tasks blocked:\s+3\b", output)


def test_summary_prints_not_run_when_no_test_ever_ran(
    runner: CliRunner, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    """A failed run that never executed a test is reported as NOT RUN."""
    FakeOrchestrator.result = _make_result(passed=False, skipped=False, tests_run=0)

    output = runner.invoke(main, ["start", "build a thing"]).output

    assert "NOT RUN" in output
    assert "no tests were executed" in output
    assert "FAILED - tests did not pass." not in output


def test_summary_prints_failed_when_a_test_ran_and_failed(
    runner: CliRunner, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    """A failed run that executed a test is reported as FAILED."""
    FakeOrchestrator.result = _make_result(passed=False, skipped=False, tests_run=1)

    output = runner.invoke(main, ["start", "build a thing"]).output

    assert "FAILED - tests did not pass." in output
    assert "NOT RUN" not in output


def test_summary_handles_a_result_without_the_new_fields(
    runner: CliRunner, fake_orchestrator: type[FakeOrchestrator]
) -> None:
    """A legacy result lacking the new fields still prints a sane summary."""
    legacy = SimpleNamespace(
        passed=True,
        cycles=1,
        tasks_completed=1,
        tasks_failed=0,
        summary="",
        stopped_reason="goal complete",
        skipped=False,
    )
    assert not hasattr(legacy, "tasks_blocked")
    assert not hasattr(legacy, "tests_run")
    FakeOrchestrator.result = cast(RunResult, legacy)

    output = runner.invoke(main, ["start", "build a thing"]).output

    assert re.search(r"Tasks blocked:\s+0\b", output)
    assert "PASSED" in output


def test_status_shows_per_task_files_changed(
    runner: CliRunner, tmp_path: Path
) -> None:
    state = RunState(
        goal="ship it",
        original_goal="ship it",
        cycles=2,
        tasks=[
            TaskModel(
                id="t1",
                description="do the thing",
                status="completed",
                files_changed=["furrow/core/orchestrator.py", "furrow/config.py"],
            )
        ],
    )
    StateStore(tmp_path / ".furrow" / "state.json").save(state)

    result = runner.invoke(main, ["status", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "[furrow/core/orchestrator.py, furrow/config.py]" in result.output


def test_status_prints_persisted_usage(runner: CliRunner, tmp_path: Path) -> None:
    """The status command prints the usage persisted in the run state."""
    state = RunState(
        goal="ship it",
        original_goal="ship it",
        cycles=2,
        usage=Usage(input_tokens=3900, output_tokens=421, calls=12),
        tasks=[TaskModel(id="t1", description="do the thing", status="completed")],
    )
    StateStore(tmp_path / ".furrow" / "state.json").save(state)

    result = runner.invoke(main, ["status", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "ship it" in result.output
    assert "LLM usage:" in result.output
    assert "12 calls" in result.output
    assert "4321 tokens" in result.output


class RecordingClient:
    """Fake LLM client that records whether it was closed."""

    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class ClientReleasingOrchestrator:
    """Fake orchestrator that owns a client recording its own closing."""

    error: ClassVar[BaseException | None] = None
    instances: ClassVar[list[ClientReleasingOrchestrator]] = []

    def __init__(
        self,
        goal: str,
        client: Any = None,
        settings: Any = None,
        state_store: Any = None,
        event_bus: Any = None,
        resume: bool = False,
    ) -> None:
        ClientReleasingOrchestrator.instances.append(self)
        self.goal = goal
        self.client = RecordingClient()

    async def run(self) -> RunResult:
        if ClientReleasingOrchestrator.error is not None:
            raise ClientReleasingOrchestrator.error
        return RunResult(
            passed=True,
            cycles=1,
            tasks_completed=1,
            summary="done",
            stopped_reason="goal complete",
        )


@pytest.fixture
def releasing_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> type[ClientReleasingOrchestrator]:
    """Replace the orchestrator with a fake that owns a closable client."""
    ClientReleasingOrchestrator.error = None
    ClientReleasingOrchestrator.instances.clear()
    monkeypatch.setattr("furrow.cli.main.Orchestrator", ClientReleasingOrchestrator)
    return ClientReleasingOrchestrator


def test_start_releases_the_llm_client_after_a_run(
    runner: CliRunner,
    releasing_orchestrator: type[ClientReleasingOrchestrator],
    tmp_path: Path,
) -> None:
    """The CLI closes the LLM client once the run is over.

    The client holds HTTP connections: they must be released when the run
    ends, otherwise every run leaks a connection pool.
    """
    result = runner.invoke(main, ["start", "build a thing", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert len(releasing_orchestrator.instances) == 1
    assert releasing_orchestrator.instances[0].client.closed is True


def test_start_releases_the_llm_client_when_the_run_raises(
    runner: CliRunner,
    releasing_orchestrator: type[ClientReleasingOrchestrator],
    tmp_path: Path,
) -> None:
    """A failing run still releases the client: the release lives in a finally."""
    releasing_orchestrator.error = RuntimeError("boom")

    result = runner.invoke(main, ["start", "build a thing", "--workspace", str(tmp_path)])

    assert result.exit_code == 1
    assert "boom" in result.output
    assert len(releasing_orchestrator.instances) == 1
    assert releasing_orchestrator.instances[0].client.closed is True
