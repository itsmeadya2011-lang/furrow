from __future__ import annotations

import pytest
from click.testing import CliRunner

from furrow.cli.main import main
from furrow.config import RunResult


class CliFakeOrchestrator:
    instances: list[CliFakeOrchestrator] = []
    raise_keyboard_interrupt = False

    def __init__(self, *, goal: str, client, max_cycles=None, **kwargs) -> None:
        self.goal = goal
        self.client = client
        self.max_cycles = max_cycles
        self.kwargs = kwargs
        CliFakeOrchestrator.instances.append(self)

    async def run(self) -> RunResult:
        if CliFakeOrchestrator.raise_keyboard_interrupt:
            raise KeyboardInterrupt
        return RunResult(goal=self.goal, cycles=2, stopped_reason="max_cycles")


@pytest.fixture(autouse=True)
def _patch_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    CliFakeOrchestrator.instances.clear()
    monkeypatch.setattr("furrow.cli.main.Orchestrator", CliFakeOrchestrator)
    yield
    CliFakeOrchestrator.instances.clear()


def test_help_lists_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "start" in result.output
    assert "web" in result.output


def test_start_help_lists_options() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["start", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--provider" in result.output
    assert "--max-cycles" in result.output
    assert "--max-parallel-tasks" in result.output
    assert "--workspace" in result.output
    assert "--dry-run" in result.output
    assert "--planner-model" in result.output
    assert "--worker-model" in result.output
    assert "--tester-model" in result.output


def test_start_constructs_orchestrator_with_goal_and_max_cycles() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["start", "my goal", "--max-cycles", "5"])
    assert result.exit_code == 0
    assert len(CliFakeOrchestrator.instances) == 1
    orch = CliFakeOrchestrator.instances[-1]
    assert orch.goal == "my goal"
    assert orch.max_cycles == 5


def test_provider_choice_accepted_and_rejected() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["start", "goal", "--provider", "openai"])
    assert result.exit_code == 0
    assert len(CliFakeOrchestrator.instances) == 1

    result = runner.invoke(main, ["start", "goal", "--provider", "invalid"])
    assert result.exit_code != 0
    assert "is not one of" in result.output


def test_dry_run_exits_without_orchestrator() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["start", "goal", "--dry-run"])
    assert result.exit_code == 0
    assert len(CliFakeOrchestrator.instances) == 0
    assert "Resolved Configuration" in result.output


def test_start_prints_summary() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["start", "my goal"])
    assert result.exit_code == 0
    assert "Furrow Summary" in result.output
    assert "Max Cycles Reached" in result.output
    assert "Cycles: 2" in result.output


def test_keyboard_interrupt_exits_zero() -> None:
    CliFakeOrchestrator.raise_keyboard_interrupt = True
    runner = CliRunner()
    result = runner.invoke(main, ["start", "goal"])
    assert result.exit_code == 0
    assert "Furrow stopped by user" in result.output
    CliFakeOrchestrator.raise_keyboard_interrupt = False


def test_web_command_passes_host_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str | None, int | None]] = []

    def fake_run(host=None, port=None):
        calls.append((host, port))

    monkeypatch.setattr("furrow.web.server.run", fake_run)
    runner = CliRunner()
    result = runner.invoke(main, ["web", "--host", "127.0.0.1", "--port", "9000"])
    assert result.exit_code == 0
    assert calls == [("127.0.0.1", 9000)]
