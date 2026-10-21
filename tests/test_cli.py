"""Tests for the Furrow command line interface.

Nothing here may touch the network or start a real run: the orchestrator is
always replaced by :class:`FakeOrchestrator`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

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
