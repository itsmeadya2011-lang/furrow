import pytest
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.agents.planner import PlannerAgent


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_initial_state():
    orch = Orchestrator(goal="do something", max_cycles=3)
    assert orch.cycles == 0
    assert orch.max_cycles == 3
    assert orch._get_tasks() == []
    assert orch._is_done() is True


def test_orchestrator_is_done_with_completed_tasks():
    orch = Orchestrator(goal="do something")
    orch._tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orch._is_done() is True


def test_orchestrator_is_done_with_failed_tasks():
    orch = Orchestrator(goal="do something")
    orch._tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert orch._is_done() is False


def test_orchestrator_is_done_with_pending_tasks():
    orch = Orchestrator(goal="do something")
    orch._tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    assert orch._is_done() is False


def test_orchestrator_build_context():
    orch = Orchestrator(goal="do something")
    assert orch._build_context() == ""
    orch._history = ["Cycle 1: Tests passed. All good."]
    context = orch._build_context()
    assert "Previous cycle results" in context
    assert "Cycle 1: Tests passed. All good." in context


def test_planner_context_passed():
    """Planner should accept context parameter without error."""
    planner = PlannerAgent()
    # We can't actually call the LLM in tests, but we can verify the method signature
    assert callable(planner.plan)
    import inspect
    sig = inspect.signature(planner.plan)
    assert "context" in sig.parameters
