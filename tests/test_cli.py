"""Tests for the ``furrow`` command line entry point.

``furrow.cli.main.Orchestrator`` is replaced with a fake for every ``start``
invocation, so these tests never construct an LLM client, never touch the network
and never write a state file outside ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result

import furrow.cli.main as cli_main
from furrow.cli.main import INTERRUPT_EXIT_CODE, INTERRUPT_MESSAGE, main
from furrow.config import settings as global_settings
from furrow.state import RunState, StateStore

runner = CliRunner()


def patch_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
    *,
    state: RunState | None = None,
    error: BaseException | None = None,
) -> list[dict[str, Any]]:
    """Swap the CLI's orchestrator for a fake; return the list of call kwargs."""
    calls: list[dict[str, Any]] = []

    class FakeOrchestrator:
        def __init__(
            self,
            goal: str,
            client: Any = None,
            settings: Any = None,
            store: Any = None,
            events: Any = None,
            console: Any = None,
        ) -> None:
            self.goal = goal
            self.settings = settings
            self.console = console
            self.stopped = False
            calls.append(
                {
                    "goal": goal,
                    "client": client,
                    "settings": settings,
                    "store": store,
                    "events": events,
                    "console": console,
                }
            )

        async def run(self) -> RunState:
            if error is not None:
                raise error
            if state is not None:
                return state
            return RunState(goal=self.goal, cycles=1, done=True)

        async def stop(self) -> None:
            self.stopped = True

    monkeypatch.setattr(cli_main, "Orchestrator", FakeOrchestrator)
    return calls


def done_state() -> RunState:
    return RunState(
        goal="build the thing",
        cycles=3,
        done=True,
        tasks=[
            {"id": "t1", "description": "first", "status": "completed", "attempts": 1},
            {"id": "t2", "description": "second", "status": "completed", "attempts": 1},
            {"id": "t3", "description": "third", "status": "failed", "attempts": 2},
        ],
        history=[
            {"cycle": 1, "passed": False, "summary": "red", "failures": ["boom"]},
            {"cycle": 2, "passed": False, "summary": "red", "failures": ["boom"]},
            {"cycle": 3, "passed": True, "summary": "green", "failures": []},
        ],
    )


# ----------------------------------------------------------------------- help


def test_main_help_exits_zero() -> None:
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "Furrow" in result.output
    for command in ("start", "status", "web"):
        assert command in result.output


def test_start_help_lists_every_option() -> None:
    result = runner.invoke(main, ["start", "--help"])

    assert result.exit_code == 0
    for option in (
        "--provider",
        "--model",
        "--planner-model",
        "--worker-model",
        "--tester-model",
        "--max-cycles",
        "--max-parallel",
        "--workspace",
        "--test-command",
        "--quiet",
        "--verbose",
    ):
        assert option in result.output, option


def test_status_help_exits_zero() -> None:
    result = runner.invoke(main, ["status", "--help"])

    assert result.exit_code == 0
    assert "--workspace" in result.output


def test_web_help_exits_zero_without_serving() -> None:
    result = runner.invoke(main, ["web", "--help"])

    assert result.exit_code == 0
    assert "--host" in result.output
    assert "--port" in result.output
    assert "--reload" in result.output


# --------------------------------------------------------------------- start


def test_start_builds_explicit_settings_and_never_touches_the_singleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = patch_orchestrator(monkeypatch, state=done_state())
    model_before = global_settings.model
    max_cycles_before = global_settings.max_cycles

    result = runner.invoke(
        main,
        ["start", "build the thing", "--model", "some-model", "--max-cycles", "1",
         "--max-parallel", "2"],
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    settings = calls[0]["settings"]
    assert calls[0]["goal"] == "build the thing"
    # The old code mutated the shared singleton (``settings.model = model``).
    assert settings is not global_settings
    assert settings.model == "some-model"
    assert settings.max_cycles == 1
    assert settings.max_parallel_tasks == 2
    assert global_settings.model == model_before
    assert global_settings.max_cycles == max_cycles_before


def test_start_overrides_do_not_leak_into_a_second_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = patch_orchestrator(monkeypatch, state=done_state())

    first = runner.invoke(main, ["start", "goal one", "--model", "model-one"])
    second = runner.invoke(main, ["start", "goal two", "--model", "model-two"])

    assert (first.exit_code, second.exit_code) == (0, 0)
    assert calls[0]["settings"].model == "model-one"
    assert calls[1]["settings"].model == "model-two"
    assert calls[0]["settings"] is not calls[1]["settings"]


def test_start_maps_every_flag_onto_the_settings_it_builds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = patch_orchestrator(monkeypatch, state=done_state())

    result = runner.invoke(
        main,
        [
            "start",
            "goal",
            "--provider",
            "openai",
            "--model",
            "m",
            "--planner-model",
            "pm",
            "--worker-model",
            "wm",
            "--tester-model",
            "tm",
            "--max-cycles",
            "7",
            "--max-parallel",
            "4",
            "--workspace",
            str(tmp_path),
            "--test-command",
            "pytest -q",
        ],
    )

    assert result.exit_code == 0
    settings = calls[0]["settings"]
    assert settings.provider == "openai"
    assert (settings.planner_model, settings.worker_model, settings.tester_model) == (
        "pm",
        "wm",
        "tm",
    )
    assert settings.max_cycles == 7
    assert settings.max_parallel_tasks == 4
    assert settings.workspace == tmp_path
    assert settings.test_command == "pytest -q"


def test_start_exit_zero_and_summary_on_a_finished_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_orchestrator(monkeypatch, state=done_state())

    result = runner.invoke(main, ["start", "build the thing", "--max-cycles", "1"])

    assert result.exit_code == 0
    assert "Goal:" in result.output
    assert "build the thing" in result.output
    assert "Cycles run: 3" in result.output
    assert "Tasks: completed=2, failed=1" in result.output
    assert "Tests: 1 passed, 2 failed" in result.output
    assert "Done: True" in result.output


def test_start_exit_one_when_the_run_halts(monkeypatch: pytest.MonkeyPatch) -> None:
    halted = done_state()
    halted.done = False
    halted.halted_reason = "max_cycles reached"
    patch_orchestrator(monkeypatch, state=halted)

    result = runner.invoke(main, ["start", "build the thing", "--max-cycles", "1"])

    assert result.exit_code == 1
    assert "Halted:" in result.output
    assert "max_cycles reached" in result.output
    assert "Done: False" in result.output


def test_start_exit_one_when_the_orchestrator_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_orchestrator(monkeypatch, error=RuntimeError("no api key configured"))

    result = runner.invoke(main, ["start", "build the thing"])

    assert result.exit_code == 1
    assert "Run failed" in result.output
    assert "no api key configured" in result.output


def test_start_exit_130_on_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_orchestrator(monkeypatch, error=KeyboardInterrupt)

    result = runner.invoke(main, ["start", "build the thing"])

    assert result.exit_code == INTERRUPT_EXIT_CODE == 130
    assert INTERRUPT_MESSAGE in result.output


def test_quiet_runs_without_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_orchestrator(monkeypatch, state=done_state())

    result = runner.invoke(main, ["start", "build the thing", "--quiet"])

    assert result.exit_code == 0
    assert calls[0]["console"] is not None
    assert calls[0]["console"].quiet is True


def test_quiet_wins_over_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_orchestrator(monkeypatch, state=done_state())

    result = runner.invoke(main, ["start", "build the thing", "--quiet", "--verbose"])

    assert result.exit_code == 0
    # The console already swallows the panels, so --verbose must not re-enable them.
    assert calls[0]["settings"].verbose is False
    assert calls[0]["console"].quiet is True


def test_verbose_alone_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_orchestrator(monkeypatch, state=done_state())

    result = runner.invoke(main, ["start", "build the thing", "--verbose"])

    assert result.exit_code == 0
    assert calls[0]["settings"].verbose is True


def test_start_prompts_for_a_missing_goal(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_orchestrator(monkeypatch, state=done_state())

    result = runner.invoke(main, ["start"], input="typed goal\n")

    assert result.exit_code == 0
    assert calls[0]["goal"] == "typed goal"


# --------------------------------------------------------------------- status


def test_status_without_saved_state_exits_zero(tmp_path: Path) -> None:
    result = runner.invoke(main, ["status", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "No saved state" in result.output
    assert str(tmp_path) in result.output


def test_status_reports_a_corrupt_state_file_as_no_state(tmp_path: Path) -> None:
    path = tmp_path / ".furrow" / "state.json"
    path.parent.mkdir(parents=True)
    path.write_text("}}} not json {{{", encoding="utf-8")

    result = runner.invoke(main, ["status", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "No saved state" in result.output


def test_status_prints_goal_and_cycle_count_from_a_saved_state(tmp_path: Path) -> None:
    StateStore(tmp_path / ".furrow" / "state.json").save(done_state())

    result: Result = runner.invoke(main, ["status", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "Goal:" in result.output
    assert "build the thing" in result.output
    assert "Cycles run: 3" in result.output
    assert "Tasks: completed=2, failed=1" in result.output
    assert "Tests: 1 passed, 2 failed" in result.output
    assert "t1" in result.output
    assert "cycle 3 green" in result.output


def test_status_of_a_halted_run_shows_the_reason(tmp_path: Path) -> None:
    halted = done_state()
    halted.done = False
    halted.halted_reason = "no progress on repeated test failures"
    StateStore(tmp_path / ".furrow" / "state.json").save(halted)

    result = runner.invoke(main, ["status", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "Halted:" in result.output
    assert "no progress on repeated test failures" in result.output
    assert "Done: False" in result.output