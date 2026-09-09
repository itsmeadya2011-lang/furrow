"""Tests for configuration models and settings."""

import pytest
from furrow.config import (
    Plan,
    Provider,
    Settings,
    TaskModel,
    TaskStatus,
    TestResult,
)


def test_provider_enum():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


def test_task_status_enum():
    assert TaskStatus.PENDING == "pending"
    assert TaskStatus.IN_PROGRESS == "in_progress"
    assert TaskStatus.COMPLETED == "completed"
    assert TaskStatus.FAILED == "failed"
    assert TaskStatus.BLOCKED == "blocked"


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"
    assert p.rationale == "ok"
    assert len(p.tasks) == 1


def test_task_model_defaults():
    t = TaskModel(id="1", description="test task")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_task_model_status_transitions():
    t = TaskModel(id="1", description="test task")
    t.mark_in_progress()
    assert t.status == TaskStatus.IN_PROGRESS

    t.mark_completed("done")
    assert t.status == TaskStatus.COMPLETED
    assert t.result == "done"

    t2 = TaskModel(id="2", description="fail task")
    t2.mark_failed("error")
    assert t2.status == TaskStatus.FAILED
    assert t2.result == "error"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True
    assert t.summary == "ok"


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="2 failures", failures=["fail1", "fail2"])
    assert t.passed is False
    assert len(t.failures) == 2


def test_settings_defaults():
    s = Settings(
        anthropic_api_key="test-key",
        provider=Provider.ANTHROPIC,
    )
    assert s.provider == Provider.ANTHROPIC
    assert s.max_cycles == 0
    assert s.max_parallel_tasks == 5
    assert s.ollama_model == "llama3.2"
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.completion_timeout == 120
    assert s.test_timeout == 120


def test_settings_ollama_provider():
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:8080")
    assert s.provider == Provider.OLLAMA
    assert s.ollama_base_url == "http://localhost:8080"
