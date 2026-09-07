from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


# --- Plan validation ---


def test_plan_with_multiple_tasks_and_dependencies():
    tasks = [
        TaskModel(id="1", description="setup", dependencies=[]),
        TaskModel(id="2", description="implement", dependencies=["1"]),
        TaskModel(id="3", description="test", dependencies=["2"]),
    ]
    plan = Plan(tasks=tasks, rationale="sequential build")
    assert len(plan.tasks) == 3
    assert plan.tasks[1].dependencies == ["1"]
    assert plan.rationale == "sequential build"


def test_plan_with_empty_tasks_list():
    plan = Plan(tasks=[], rationale="nothing to do")
    assert plan.tasks == []
    assert plan.rationale == "nothing to do"


# --- TaskModel defaults ---


def test_task_status_defaults_to_pending():
    task = TaskModel(id="1", description="do thing")
    assert task.status == "pending"


def test_task_dependencies_defaults_to_empty_list():
    task = TaskModel(id="1", description="do thing")
    assert task.dependencies == []


def test_task_files_defaults_to_empty_list():
    task = TaskModel(id="1", description="do thing")
    assert task.files == []


# --- TestResult behavior ---


def test_test_result_passing():
    result = TestResult(passed=True, summary="all good", failures=[])
    assert result.passed is True
    assert result.summary == "all good"


def test_test_result_failing_with_multiple_failures():
    failures = ["module A import failed", "test B assertion failed"]
    result = TestResult(passed=False, summary="2 failures", failures=failures)
    assert result.passed is False
    assert len(result.failures) == 2
    assert result.failures[0] == "module A import failed"


def test_test_result_failures_defaults_to_empty_list():
    result = TestResult(passed=True, summary="ok")
    assert result.failures == []


# --- Settings defaults ---


def test_settings_default_provider_is_anthropic():
    settings = Settings()
    assert settings.provider == Provider.ANTHROPIC


def test_settings_default_model_values():
    settings = Settings()
    assert settings.model == "claude-sonnet-4-20250514"
    assert settings.planner_model == "claude-3-5-haiku-20241022"
    assert settings.worker_model == "claude-3-5-sonnet-20241022"
    assert settings.tester_model == "claude-3-5-sonnet-20241022"


def test_settings_max_parallel_tasks_defaults_to_five():
    settings = Settings()
    assert settings.max_parallel_tasks == 5


def test_settings_workspace_defaults_to_current_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    assert settings.workspace == tmp_path


# --- LLMClient initialization ---


def test_llm_client_initializes_with_default_settings():
    client = LLMClient()
    assert client.settings is not None
    assert client._anthropic is None
    assert client._openai is None


def test_llm_client_anthropic_raises_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings(anthropic_api_key=None)
    client = LLMClient(settings=settings)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        _ = client.anthropic


def test_llm_client_openai_raises_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(openai_api_key=None)
    client = LLMClient(settings=settings)
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
        _ = client.openai


# --- Orchestrator logic ---


def test_is_done_returns_true_when_all_tasks_completed():
    orchestrator = Orchestrator(goal="test")
    completed_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    with patch.object(orchestrator, "_get_tasks", return_value=completed_tasks):
        assert orchestrator._is_done() is True


def test_is_done_returns_false_when_tasks_are_pending():
    orchestrator = Orchestrator(goal="test")
    pending_tasks = [
        TaskModel(id="1", description="a", status="pending"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    with patch.object(orchestrator, "_get_tasks", return_value=pending_tasks):
        assert orchestrator._is_done() is False


def test_is_done_returns_false_when_tasks_failed():
    orchestrator = Orchestrator(goal="test")
    failed_tasks = [
        TaskModel(id="1", description="a", status="failed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    with patch.object(orchestrator, "_get_tasks", return_value=failed_tasks):
        assert orchestrator._is_done() is False


def test_get_tasks_returns_empty_list_when_no_plan_set():
    orchestrator = Orchestrator(goal="test")
    assert orchestrator._get_tasks() == []


def test_is_done_returns_true_when_no_tasks_at_all():
    orchestrator = Orchestrator(goal="test")
    with patch.object(orchestrator, "_get_tasks", return_value=[]):
        assert orchestrator._is_done() is True
