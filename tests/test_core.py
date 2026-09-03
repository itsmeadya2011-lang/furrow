import pytest
from furrow.config import Plan, TaskModel, TestResult, Provider, Settings, override_settings


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_is_done_no_tasks():
    from furrow.core.orchestrator import Orchestrator
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.goal = "test"
    orchestrator.client = mock_client
    orchestrator.cycles = 0
    orchestrator._current_plan = None
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_with_tasks():
    from furrow.core.orchestrator import Orchestrator
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.goal = "test"
    orchestrator.client = mock_client
    orchestrator.cycles = 0
    plan = Plan(tasks=[
        TaskModel(id="1", description="a"),
        TaskModel(id="2", description="b"),
    ], rationale="ok")
    orchestrator._current_plan = plan

    assert orchestrator._is_done() is False

    for t in plan.tasks:
        t.status = "completed"
    assert orchestrator._is_done() is True

    plan.tasks[0].status = "failed"
    assert orchestrator._is_done() is False


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_cycles == 0
    assert s.max_parallel_tasks == 5
    assert s.ollama_base_url == "http://localhost:11434"


def test_provider_enum():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


def test_override_settings():
    s = Settings()
    with override_settings(max_cycles=5):
        assert s.max_cycles == 5
    assert s.max_cycles == 0
