"""Tests for furrow.config — data models and settings."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult

# ------------------------------------------------------------------ #
# TaskModel
# ------------------------------------------------------------------ #

def test_task_model_defaults():
    t = TaskModel(id="1", description="do something")
    assert t.id == "1"
    assert t.description == "do something"
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_task_model_with_values():
    t = TaskModel(
        id="2",
        description="do more",
        files=["a.py", "b.py"],
        dependencies=["1"],
        status="completed",
        result="done",
    )
    assert t.files == ["a.py", "b.py"]
    assert t.dependencies == ["1"]
    assert t.status == "completed"
    assert t.result == "done"


def test_task_model_requires_id_and_description():
    with pytest.raises(ValidationError):
        TaskModel(description="no id")
    with pytest.raises(ValidationError):
        TaskModel(id="1")


# ------------------------------------------------------------------ #
# Plan
# ------------------------------------------------------------------ #

def test_plan_requires_tasks_and_rationale():
    with pytest.raises(ValidationError):
        Plan(tasks=[])
    with pytest.raises(ValidationError):
        Plan(tasks=[TaskModel(id="1", description="x")])
    p = Plan(tasks=[TaskModel(id="1", description="x")], rationale="ok")
    assert len(p.tasks) == 1
    assert p.rationale == "ok"


def test_plan_serialization():
    p = Plan(tasks=[TaskModel(id="1", description="x")], rationale="ok")
    dumped = p.model_dump()
    assert dumped["tasks"][0]["id"] == "1"
    assert dumped["rationale"] == "ok"


# ------------------------------------------------------------------ #
# TestResult
# ------------------------------------------------------------------ #

def test_test_result_defaults():
    t = TestResult(passed=True, summary="ok")
    assert t.failures == []


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="fail", failures=["err1", "err2"])
    assert t.passed is False
    assert t.failures == ["err1", "err2"]


# ------------------------------------------------------------------ #
# Provider enum
# ------------------------------------------------------------------ #

def test_provider_values():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"


# ------------------------------------------------------------------ #
# Settings
# ------------------------------------------------------------------ #

def test_settings_defaults(monkeypatch: pytest.MonkeyPatch):
    # Clear env vars that could interfere.
    monkeypatch.delenv("FURROW_PROVIDER", raising=False)
    monkeypatch.delenv("FURROW_MODEL", raising=False)
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.workspace == Path.cwd()
    assert s.log_level == "INFO"


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FURROW_PROVIDER", "openai")
    monkeypatch.setenv("FURROW_MAX_CYCLES", "10")
    s = Settings()
    assert s.provider == Provider.OPENAI
    assert s.max_cycles == 10
