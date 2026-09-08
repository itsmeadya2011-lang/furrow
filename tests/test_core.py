import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_is_done_no_tasks():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="test")
    assert orch._is_done() is False


def test_orchestrator_is_done_all_completed():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="test")
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orch._is_done() is True


def test_orchestrator_is_done_has_failed():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="test")
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert orch._is_done() is False


def test_orchestrator_is_done_max_cycles():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="test")
    orch.cycles = 5
    orch.tasks = [TaskModel(id="1", description="a", status="completed")]
    orch.client = type("FakeClient", (), {"settings": Settings(max_cycles=5)})()
    assert orch._is_done() is True


def test_tester_detect_runners_python(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest]")
    agent = TesterAgent(settings=Settings(workspace=tmp_path))
    cmds = agent._detect_test_runners()
    assert any("pytest" in cmd for cmd in cmds)


def test_tester_detect_runners_node(tmp_path):
    (tmp_path / "package.json").write_text("{}")
    agent = TesterAgent(settings=Settings(workspace=tmp_path))
    cmds = agent._detect_test_runners()
    assert any("npm" in cmd for cmd in cmds)


def test_tester_detect_runners_rust(tmp_path):
    (tmp_path / "Cargo.toml").write_text("[package]")
    agent = TesterAgent(settings=Settings(workspace=tmp_path))
    cmds = agent._detect_test_runners()
    assert any("cargo" in cmd for cmd in cmds)


def test_tester_detect_runners_go(tmp_path):
    (tmp_path / "go.mod").write_text("module test")
    agent = TesterAgent(settings=Settings(workspace=tmp_path))
    cmds = agent._detect_test_runners()
    assert any("go" in cmd for cmd in cmds)


def test_tester_detect_runners_none(tmp_path):
    agent = TesterAgent(settings=Settings(workspace=tmp_path))
    cmds = agent._detect_test_runners()
    assert cmds == []


@pytest.mark.asyncio
async def test_tester_run_success():
    agent = TesterAgent()
    with patch.object(agent, "_run_tests", new_callable=AsyncMock, return_value="all good"):
        result = await agent.run("goal", [])
        assert result.passed is True


@pytest.mark.asyncio
async def test_tester_run_failure():
    fake_client = type("FakeClient", (), {
        "settings": Settings(),
        "complete": AsyncMock(return_value=json.dumps({"passed": False, "summary": "bad", "failures": ["x"]})),
    })()
    agent = TesterAgent(client=fake_client)
    with patch.object(agent, "_run_tests", new_callable=AsyncMock, return_value="FAILURES"):
        result = await agent.run("goal", [])
        assert result.passed is False
        assert result.failures == ["x"]


def test_worker_format_project_structure(tmp_path):
    (tmp_path / "a.py").write_text("pass")
    (tmp_path / "b.py").write_text("pass")
    task = TaskModel(id="1", description="do thing", files=["a.py"])
    agent = WorkerAgent(task=task, settings=Settings(workspace=tmp_path))
    structure = agent._format_project_structure()
    assert "a.py" in structure
    assert "b.py" in structure


def test_planner_format_project_structure(tmp_path):
    (tmp_path / "main.py").write_text("pass")
    agent = PlannerAgent(settings=Settings(workspace=tmp_path))
    structure = agent._format_project_structure()
    assert "main.py" in structure
