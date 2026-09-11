import inspect
from pathlib import Path

import pytest
from click.testing import CliRunner

from furrow.agents.tester import TesterAgent
from furrow.cli.main import main as cli_main
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_is_done_no_tasks():
    orch = Orchestrator.__new__(Orchestrator)
    orch.tasks = []
    orch.cycles = 0
    orch.settings = Settings()
    orch.last_plan_had_no_tasks = False
    assert orch._is_done() is False


def test_orchestrator_is_done_all_completed():
    orch = Orchestrator.__new__(Orchestrator)
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    orch.cycles = 1
    orch.settings = Settings(max_cycles=0)
    orch.last_plan_had_no_tasks = True
    assert orch._is_done() is True


def test_orchestrator_is_done_has_failed():
    orch = Orchestrator.__new__(Orchestrator)
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    orch.cycles = 1
    orch.settings = Settings(max_cycles=0)
    orch.last_plan_had_no_tasks = True
    assert orch._is_done() is False


def test_orchestrator_is_done_max_cycles():
    orch = Orchestrator.__new__(Orchestrator)
    orch.tasks = [TaskModel(id="1", description="a", status="completed")]
    orch.cycles = 5
    orch.settings = Settings(max_cycles=5)
    orch.last_plan_had_no_tasks = False
    assert orch._is_done() is True


def test_orchestrator_is_done_not_all_completed():
    orch = Orchestrator.__new__(Orchestrator)
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    orch.cycles = 1
    orch.settings = Settings(max_cycles=0)
    orch.last_plan_had_no_tasks = True
    assert orch._is_done() is False


def test_orchestrator_is_done_plan_has_tasks_not_completed():
    orch = Orchestrator.__new__(Orchestrator)
    orch.tasks = [TaskModel(id="1", description="a", status="pending")]
    orch.cycles = 1
    orch.settings = Settings(max_cycles=0)
    orch.last_plan_had_no_tasks = False
    assert orch._is_done() is False


def test_tester_detects_pyproject(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    tester = TesterAgent.__new__(TesterAgent)
    tester.workspace = tmp_path
    cmds = tester._detect_test_commands()
    assert any("pytest" in " ".join(c) for c in cmds)


def test_tester_detects_package_json(tmp_path: Path):
    (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}')
    tester = TesterAgent.__new__(TesterAgent)
    tester.workspace = tmp_path
    cmds = tester._detect_test_commands()
    assert any("npm test" in " ".join(c) for c in cmds)


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli_main, ["--version"])
    assert result.exit_code == 0


def test_cli_workspace_invalid():
    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        ["start", "--workspace", "/nonexistent/path/12345", "do something"],
    )
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_llm_client_has_retry():
    assert "retry" in inspect.getsource(LLMClient.complete)
