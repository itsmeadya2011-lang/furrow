from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class _StubClient:
    pass


def test_orchestrator_initial_state():
    orch = Orchestrator(goal="build a thing", client=_StubClient())  # type: ignore[arg-type]
    assert orch.goal == "build a thing"
    assert orch.cycles == 0
    assert orch._last_test_passed is False
    assert orch._failure_context is None
    assert orch._get_tasks() == []


def test_orchestrator_is_done_false_initially():
    orch = Orchestrator(goal="x", client=_StubClient())  # type: ignore[arg-type]
    assert orch._is_done() is False


def test_orchestrator_is_done_after_pass():
    orch = Orchestrator(goal="x", client=_StubClient())  # type: ignore[arg-type]
    orch._last_test_passed = True
    assert orch._is_done() is True


def test_orchestrator_get_tasks_after_plan():
    orch = Orchestrator(goal="x", client=_StubClient())  # type: ignore[arg-type]
    orch._current_plan = Plan(tasks=[TaskModel(id="1", description="a")], rationale="r")
    assert len(orch._get_tasks()) == 1
    assert orch._get_tasks()[0].id == "1"


def test_orchestrator_goal_not_overwritten_on_failure():
    orch = Orchestrator(goal="original", client=_StubClient())  # type: ignore[arg-type]
    orch._failure_context = "tests broken"
    assert orch.goal == "original"
    assert orch._failure_context == "tests broken"
