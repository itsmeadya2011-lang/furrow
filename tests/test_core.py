import json
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from furrow.config import LLMClient, Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_is_done_no_tasks():
    orchestrator = Orchestrator(goal="test")
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_all_completed():
    orchestrator = Orchestrator(goal="test")
    tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    with patch.object(orchestrator, "_get_tasks", return_value=tasks):
        assert orchestrator._is_done() is True


def test_orchestrator_is_done_any_failed():
    orchestrator = Orchestrator(goal="test")
    tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    with patch.object(orchestrator, "_get_tasks", return_value=tasks):
        assert orchestrator._is_done() is False


def test_orchestrator_is_done_mixed_status():
    orchestrator = Orchestrator(goal="test")
    tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    with patch.object(orchestrator, "_get_tasks", return_value=tasks):
        assert orchestrator._is_done() is False


def test_llm_client_unsupported_provider():
    settings = Settings()
    client = LLMClient(settings=settings)
    client.settings.provider = "invalid_provider"
    with pytest.raises(ValueError, match="Unsupported provider"):
        import asyncio
        asyncio.run(client.complete("hello"))


def test_settings_defaults():
    settings = Settings()
    assert settings.provider == Provider.ANTHROPIC
    assert settings.model == "claude-sonnet-4-20250514"
    assert settings.planner_model == "claude-3-5-haiku-20241022"
    assert settings.worker_model == "claude-3-5-sonnet-20241022"
    assert settings.tester_model == "claude-3-5-sonnet-20241022"
    assert settings.anthropic_api_key is None
    assert settings.openai_api_key is None
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.max_parallel_tasks == 5
    assert settings.max_cycles == 0
    assert settings.log_level == "INFO"


def test_task_model_json_round_trip():
    task = TaskModel(id="1", description="do thing", files=["a.py"], dependencies=["0"])
    data = task.model_dump_json()
    restored = TaskModel.model_validate_json(data)
    assert restored.id == "1"
    assert restored.description == "do thing"
    assert restored.files == ["a.py"]
    assert restored.dependencies == ["0"]
    assert restored.status == "pending"
    assert restored.result is None


def test_plan_validation_missing_tasks():
    with pytest.raises(ValidationError):
        Plan(rationale="ok")


def test_plan_validation_missing_rationale():
    with pytest.raises(ValidationError):
        Plan(tasks=[TaskModel(id="1", description="x")])


def test_task_model_validation_missing_id():
    with pytest.raises(ValidationError):
        TaskModel(description="do thing")


def test_task_model_validation_missing_description():
    with pytest.raises(ValidationError):
        TaskModel(id="1")
