"""Unit tests for furrow.config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from furrow.config import (
    Plan,
    Provider,
    Settings,
    TaskModel,
    TestResult,
)


def test_provider_members() -> None:
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"


def test_settings_defaults() -> None:
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"
    assert s.ollama_base_url == "http://localhost:11434"


def test_settings_validation_max_parallel_tasks() -> None:
    with pytest.raises(ValidationError):
        Settings(max_parallel_tasks=0)
    with pytest.raises(ValidationError):
        Settings(max_parallel_tasks=-1)


def test_settings_validation_max_cycles() -> None:
    with pytest.raises(ValidationError):
        Settings(max_cycles=-1)


def test_task_model_defaults() -> None:
    t = TaskModel(id="1", description="d")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_task_model_separate_defaults() -> None:
    a = TaskModel(id="1", description="d")
    b = TaskModel(id="2", description="d")
    assert a.files is not b.files
    assert a.dependencies is not b.dependencies


def test_plan_from_dict() -> None:
    data = {
        "tasks": [{"id": "1", "description": "do thing", "files": ["a.py"], "dependencies": []}],
        "rationale": "because",
    }
    plan = Plan(**data)
    assert plan.rationale == "because"
    assert plan.tasks[0].id == "1"
    assert plan.tasks[0].files == ["a.py"]


def test_test_result_defaults() -> None:
    r = TestResult(passed=True, summary="ok")
    assert r.passed is True
    assert r.summary == "ok"
    assert r.failures == []


def test_test_result_not_collected() -> None:
    assert TestResult.__test__ is False
