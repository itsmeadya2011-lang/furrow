import pytest
from furrow.agents.tester import TesterAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_initialization():
    orch = Orchestrator(goal="test goal")
    assert orch.goal == "test goal"
    assert orch.cycles == 0
    assert orch.max_cycles == 0
    assert orch._get_tasks() == []


def test_orchestrator_max_cycles_parameter():
    orch = Orchestrator(goal="test goal", max_cycles=5)
    assert orch.max_cycles == 5


def test_orchestrator_get_tasks_with_plan():
    orch = Orchestrator(goal="test goal")
    plan = Plan(
        tasks=[TaskModel(id="1", description="task one")],
        rationale="plan",
    )
    orch._current_plan = plan
    tasks = orch._get_tasks()
    assert len(tasks) == 1
    assert tasks[0].id == "1"


def test_orchestrator_is_done_empty_tasks():
    orch = Orchestrator(goal="test goal")
    assert orch._is_done() is True


def test_orchestrator_is_done_all_completed():
    orch = Orchestrator(goal="test goal")
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="completed"),
        ],
        rationale="plan",
    )
    orch._current_plan = plan
    assert orch._is_done() is True


def test_orchestrator_is_done_some_failed():
    orch = Orchestrator(goal="test goal")
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="failed"),
        ],
        rationale="plan",
    )
    orch._current_plan = plan
    assert orch._is_done() is False


def test_orchestrator_is_done_mixed_status():
    orch = Orchestrator(goal="test goal")
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="pending"),
        ],
        rationale="plan",
    )
    orch._current_plan = plan
    assert orch._is_done() is False


def test_tester_detects_python_project(tmp_path, monkeypatch):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[project]\nname = 'test'\n")
    tester = TesterAgent.__new__(TesterAgent)
    tester.client = LLMClient.__new__(LLMClient)
    tester.client.settings = LLMClient().settings
    tester.client.settings = tester.client.settings.model_copy(update={"workspace": tmp_path})
    cmds = tester._detect_test_commands()
    assert any("pytest" in cmd for cmd in cmds)


def test_tester_detects_node_project(tmp_path):
    pkg = tmp_path / "package.json"
    pkg.write_text("{}")
    tester = TesterAgent.__new__(TesterAgent)
    tester.client = LLMClient.__new__(LLMClient)
    tester.client.settings = LLMClient().settings
    tester.client.settings = tester.client.settings.model_copy(update={"workspace": tmp_path})
    cmds = tester._detect_test_commands()
    assert any("npm" in cmd for cmd in cmds)


def test_tester_detects_rust_project(tmp_path):
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text("[package]\nname = 'test'\n")
    tester = TesterAgent.__new__(TesterAgent)
    tester.client = LLMClient.__new__(LLMClient)
    tester.client.settings = LLMClient().settings
    tester.client.settings = tester.client.settings.model_copy(update={"workspace": tmp_path})
    cmds = tester._detect_test_commands()
    assert any("cargo" in cmd for cmd in cmds)


def test_settings_max_tokens_default():
    from furrow.config import settings
    assert settings.max_tokens == 4096


def test_settings_max_cycles_default():
    from furrow.config import settings
    assert settings.max_cycles == 0
