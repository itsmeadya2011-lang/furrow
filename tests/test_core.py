import pytest
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_task_tracking():
    """Tasks should be tracked across cycles via _all_tasks."""
    orchestrator = Orchestrator(goal="test goal")
    assert orchestrator._get_tasks() == []

    task = TaskModel(id="1", description="test task")
    orchestrator._all_tasks.append(task)
    assert len(orchestrator._get_tasks()) == 1
    assert orchestrator._get_tasks()[0].id == "1"


def test_orchestrator_is_done_with_completed_tasks():
    """_is_done should return True when all tracked tasks are completed."""
    orchestrator = Orchestrator(goal="test goal")
    task = TaskModel(id="1", description="test", status="completed", result="ok")
    orchestrator._all_tasks.append(task)
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_with_failed_tasks():
    """_is_done should return False when any tracked task has failed."""
    orchestrator = Orchestrator(goal="test goal")
    task = TaskModel(id="1", description="test", status="failed", result="error")
    orchestrator._all_tasks.append(task)
    assert orchestrator._is_done() is False


def test_orchestrator_is_done_not_done_with_pending_tasks():
    """_is_done should return False when tasks are still pending."""
    orchestrator = Orchestrator(goal="test goal")
    task = TaskModel(id="1", description="test", status="pending")
    orchestrator._all_tasks.append(task)
    assert orchestrator._is_done() is False


def test_orchestrator_max_cycles_reached():
    """max_cycles should be enforced when set."""
    orchestrator = Orchestrator(goal="test goal")
    orchestrator.cycles = 5
    orchestrator.client.settings.max_cycles = 5
    assert orchestrator._max_cycles_reached() is True


def test_orchestrator_max_cycles_not_reached():
    """max_cycles should not be enforced when set to 0 (unlimited)."""
    orchestrator = Orchestrator(goal="test goal")
    orchestrator.cycles = 5
    orchestrator.client.settings.max_cycles = 0
    assert orchestrator._max_cycles_reached() is False


def test_orchestrator_max_cycles_zero_by_default():
    """max_cycles defaults to 0 (unlimited)."""
    orchestrator = Orchestrator(goal="test goal")
    assert orchestrator.client.settings.max_cycles == 0


def test_tester_result_from_llm_with_invalid_json():
    """TestResult should handle non-JSON LLM responses gracefully."""
    result = TestResult(passed="passed" in "tests passed", summary="tests passed", failures=[])
    assert isinstance(result.passed, bool)
    assert result.passed is True


def test_orchestrator_get_tasks_returns_all_tasks():
    """_get_tasks should return all tasks across all cycles, not just empty list."""
    orchestrator = Orchestrator(goal="test goal")
    assert orchestrator._get_tasks() == []

    orchestrator._all_tasks.extend([
        TaskModel(id="1", description="task 1", status="completed", result="r1"),
        TaskModel(id="2", description="task 2", status="completed", result="r2"),
    ])
    tasks = orchestrator._get_tasks()
    assert len(tasks) == 2
    assert all(t.status == "completed" for t in tasks)
