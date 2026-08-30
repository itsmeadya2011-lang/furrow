import pytest
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_get_tasks_returns_current_plan_tasks():
    client = LLMClient()
    plan = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    orchestrator = Orchestrator(goal="test", client=client)
    orchestrator.current_plan = plan
    assert orchestrator._get_tasks() == plan.tasks


def test_get_tasks_returns_empty_when_no_plan():
    client = LLMClient()
    orchestrator = Orchestrator(goal="test", client=client)
    assert orchestrator._get_tasks() == []


def test_is_done_true_when_all_completed_no_failures():
    client = LLMClient()
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="completed"),
        ],
        rationale="ok",
    )
    orchestrator = Orchestrator(goal="test", client=client, max_cycles=5)
    orchestrator.current_plan = plan
    orchestrator.cycles = 1
    assert orchestrator._is_done() is True


def test_is_done_false_when_there_are_failures():
    client = LLMClient()
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="failed"),
        ],
        rationale="ok",
    )
    orchestrator = Orchestrator(goal="test", client=client, max_cycles=5)
    orchestrator.current_plan = plan
    orchestrator.cycles = 1
    assert orchestrator._is_done() is False


def test_is_done_true_when_max_cycles_is_reached():
    client = LLMClient()
    plan = Plan(
        tasks=[TaskModel(id="1", description="task 1", status="pending")],
        rationale="ok",
    )
    orchestrator = Orchestrator(goal="test", client=client, max_cycles=2)
    orchestrator.current_plan = plan
    orchestrator.cycles = 2
    assert orchestrator._is_done() is True


def test_init_accepts_max_cycles():
    client = LLMClient()
    orchestrator = Orchestrator(goal="test", client=client, max_cycles=10)
    assert orchestrator.max_cycles == 10


def test_settings_has_new_fields():
    s = Settings(request_timeout=60, retry_attempts=5, retry_backoff=2.0)
    assert s.request_timeout == 60
    assert s.retry_attempts == 5
    assert s.retry_backoff == 2.0
