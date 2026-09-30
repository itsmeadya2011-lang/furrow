"""Tests for furrow.config — settings, models, and provider configuration."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult

# ---------------------------------------------------------------------------
#  Provider enum
# ---------------------------------------------------------------------------

def test_provider_values():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


def test_provider_from_env():
    assert Provider("openai") == Provider.OPENAI
    assert Provider("anthropic") == Provider.ANTHROPIC
    assert Provider("ollama") == Provider.OLLAMA


# ---------------------------------------------------------------------------
#  Settings
# ---------------------------------------------------------------------------

def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.llm_timeout == 120.0
    assert s.test_timeout == 120.0
    assert s.ollama_model == "llama3.1"


def test_settings_env_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FURROW_PROVIDER", "openai")
    monkeypatch.setenv("FURROW_MAX_CYCLES", "10")
    s = Settings()
    assert s.provider == Provider.OPENAI
    assert s.max_cycles == 10


def test_settings_workspace_default():
    s = Settings()
    assert s.workspace == Path.cwd()


# ---------------------------------------------------------------------------
#  TaskModel
# ---------------------------------------------------------------------------

def test_task_model_defaults():
    t = TaskModel(id="t1", description="Do something")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_task_model_with_fields():
    t = TaskModel(
        id="t1",
        description="Add auth",
        files=["src/auth.py"],
        dependencies=["t0"],
        status="completed",
        result="Done",
    )
    assert t.files == ["src/auth.py"]
    assert t.dependencies == ["t0"]
    assert t.status == "completed"


# ---------------------------------------------------------------------------
#  Plan
# ---------------------------------------------------------------------------

def test_plan_parse():
    p = Plan(
        tasks=[TaskModel(id="1", description="do thing")],
        rationale="ok",
    )
    assert p.tasks[0].description == "do thing"
    assert p.rationale == "ok"


def test_plan_from_dict():
    data = {
        "tasks": [
            {"id": "1", "description": "task 1", "files": ["a.py"], "dependencies": []},
            {"id": "2", "description": "task 2", "files": ["b.py"], "dependencies": ["1"]},
        ],
        "rationale": "A multi-task plan",
    }
    plan = Plan(**data)
    assert len(plan.tasks) == 2
    assert plan.tasks[1].dependencies == ["1"]


def test_plan_requires_tasks():
    with pytest.raises(ValidationError):
        Plan(rationale="missing tasks")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
#  TestResult
# ---------------------------------------------------------------------------

def test_test_result_defaults():
    t = TestResult(passed=True, summary="All good")
    assert t.passed is True
    assert t.failures == []


def test_test_result_with_failures():
    t = TestResult(
        passed=False,
        summary="2 failures",
        failures=["test_a failed", "test_b failed"],
    )
    assert len(t.failures) == 2
