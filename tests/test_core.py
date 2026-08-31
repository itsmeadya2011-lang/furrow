import pytest
from furrow.config import Plan, TaskModel, TestResult, Settings, Provider
from furrow.llm import LLMClient
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.status == "pending"
    assert t.files == []
    assert t.dependencies == []
    assert t.result is None


def test_plan_multiple_tasks():
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="first task"),
            TaskModel(id="2", description="second task"),
            TaskModel(id="3", description="third task"),
        ],
        rationale="This is the plan",
    )
    assert len(plan.tasks) == 3
    assert plan.tasks[0].description == "first task"
    assert plan.tasks[2].description == "third task"
    assert all(t.status == "pending" for t in plan.tasks)


def test_test_result_with_failures():
    t = TestResult(
        passed=False,
        summary="2 failures",
        failures=["test_a failed", "test_b failed"],
    )
    assert t.passed is False
    assert t.summary == "2 failures"
    assert len(t.failures) == 2
    assert "test_a failed" in t.failures


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.model == "claude-sonnet-4-20250514"
    assert s.max_cycles == 0
    assert s.max_parallel_tasks == 5
    assert s.log_level == "INFO"


def test_llm_client_init():
    client = LLMClient()
    assert client.settings is not None
    assert client.settings.provider == Provider.ANTHROPIC
    assert client._anthropic is None
    assert client._openai is None


def test_orchestrator_init():
    orch = Orchestrator(goal="build a feature")
    assert orch.goal == "build a feature"
    assert orch.cycles == 0
    assert orch._plan is None
    assert isinstance(orch.client, LLMClient)


def test_orchestrator_is_done_empty(monkeypatch):
    orch = Orchestrator(goal="build a feature")
    monkeypatch.setattr(orch, "_get_tasks", lambda: [])
    assert orch._is_done() is True


def test_orchestrator_is_done_all_completed(monkeypatch):
    orch = Orchestrator(goal="build a feature")
    tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    monkeypatch.setattr(orch, "_get_tasks", lambda: tasks)
    assert orch._is_done() is True


def test_orchestrator_is_done_with_failed_task(monkeypatch):
    orch = Orchestrator(goal="build a feature")
    tasks = [TaskModel(id="1", description="a", status="failed")]
    monkeypatch.setattr(orch, "_get_tasks", lambda: tasks)
    assert orch._is_done() is False


def test_orchestrator_is_done_with_pending_task(monkeypatch):
    orch = Orchestrator(goal="build a feature")
    tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    monkeypatch.setattr(orch, "_get_tasks", lambda: tasks)
    assert orch._is_done() is False
