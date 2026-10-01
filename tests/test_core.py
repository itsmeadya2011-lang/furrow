from io import StringIO

from rich.console import Console

from furrow.agents.tester import TesterAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_get_tasks_returns_empty_when_no_plan():
    orch = Orchestrator(goal="test")
    assert orch._get_tasks() == []


def test_get_tasks_returns_plan_tasks():
    orch = Orchestrator(goal="test")
    tasks = [
        TaskModel(id="1", description="do thing"),
        TaskModel(id="2", description="another task"),
    ]
    orch.plan = Plan(tasks=tasks, rationale="ok")
    assert orch._get_tasks() == tasks


def test_is_done_empty_tasks():
    orch = Orchestrator(goal="test")
    assert orch._is_done() is True


def test_is_done_all_completed():
    orch = Orchestrator(goal="test")
    tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
        TaskModel(id="3", description="c", status="completed"),
    ]
    orch.plan = Plan(tasks=tasks, rationale="ok")
    assert orch._is_done() is True


def test_is_done_some_failed():
    orch = Orchestrator(goal="test")
    tasks = [
        TaskModel(id="1", description="a", status="failed"),
        TaskModel(id="2", description="b", status="completed"),
        TaskModel(id="3", description="c", status="completed"),
    ]
    orch.plan = Plan(tasks=tasks, rationale="ok")
    assert orch._is_done() is False


def test_is_done_some_pending():
    orch = Orchestrator(goal="test")
    tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
        TaskModel(id="3", description="c", status="pending"),
    ]
    orch.plan = Plan(tasks=tasks, rationale="ok")
    assert orch._is_done() is False


async def test_max_cycles_enforced(monkeypatch):
    orch = Orchestrator(goal="test")
    monkeypatch.setattr(orch.settings, "max_cycles", 1)
    orch.plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="pending"),
            TaskModel(id="2", description="b", status="pending"),
            TaskModel(id="3", description="c", status="pending"),
        ],
        rationale="ok",
    )
    cycle_calls = 0

    async def mock_cycle():
        nonlocal cycle_calls
        cycle_calls += 1

    monkeypatch.setattr(orch, "_cycle", mock_cycle)
    await orch.run()
    assert cycle_calls == 1
    assert orch.cycles == 2


def test_console_injection():
    buf = StringIO()
    console = Console(file=buf)
    orch = Orchestrator(goal="test", console=console)
    assert orch.console is console
    orch.console.print("hello world")
    assert "hello world" in buf.getvalue()


def test_tester_instance_attribute():
    orch = Orchestrator(goal="test")
    assert isinstance(orch.tester, TesterAgent)


def test_settings_from_client():
    orch = Orchestrator(goal="test")
    assert orch.settings is orch.client.settings
    assert isinstance(orch.settings, Settings)
