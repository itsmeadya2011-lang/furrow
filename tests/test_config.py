"""Tests for furrow.config module."""
from pathlib import Path

import pytest

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


# --- Provider enum ---

def test_provider_enum_values():
    """Provider enum has correct string values."""
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"


def test_provider_str_member():
    """Provider is a str-enum subclass."""
    assert isinstance(Provider.ANTHROPIC, str)
    assert Provider.ANTHROPIC == "anthropic"


# --- Settings defaults ---

def test_settings_defaults():
    """Settings has correct default values."""
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_cycles == 0
    assert s.max_parallel_tasks == 5
    assert s.log_level == "INFO"
    assert s.ollama_base_url == "http://localhost:11434"
    assert isinstance(s.workspace, Path)


def test_settings_env_override(monkeypatch):
    """Settings reads FURROW_-prefixed environment variables."""
    monkeypatch.setenv("FURROW_PROVIDER", "openai")
    monkeypatch.setenv("FURROW_MAX_CYCLES", "3")
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "2")
    monkeypatch.setenv("FURROW_MODEL", "gpt-4-turbo")
    s = Settings()
    assert s.provider == Provider.OPENAI
    assert s.max_cycles == 3
    assert s.max_parallel_tasks == 2
    assert s.model == "gpt-4-turbo"


def test_settings_ollama_provider(monkeypatch):
    """Settings can use Ollama provider."""
    monkeypatch.setenv("FURROW_PROVIDER", "ollama")
    s = Settings()
    assert s.provider == Provider.OLLAMA


# --- TaskModel ---

def test_task_model_defaults():
    """TaskModel has correct default values."""
    t = TaskModel(id="1", description="do something")
    assert t.status == "pending"
    assert t.files == []
    assert t.dependencies == []
    assert t.result is None


def test_task_model_with_values():
    """TaskModel stores all provided values."""
    t = TaskModel(
        id="2",
        description="build auth",
        files=["src/auth.py", "tests/test_auth.py"],
        dependencies=["1"],
        status="completed",
        result="done",
    )
    assert t.id == "2"
    assert t.description == "build auth"
    assert t.files == ["src/auth.py", "tests/test_auth.py"]
    assert t.dependencies == ["1"]
    assert t.status == "completed"
    assert t.result == "done"


# --- Plan ---

def test_plan_creation():
    """Plan stores tasks and rationale."""
    tasks = [TaskModel(id="1", description="task one", status="completed")]
    p = Plan(tasks=tasks, rationale="A test plan")
    assert len(p.tasks) == 1
    assert p.rationale == "A test plan"
    assert p.tasks[0].id == "1"


def test_plan_empty_tasks():
    """Plan can have no tasks."""
    p = Plan(tasks=[], rationale="Nothing to do")
    assert p.tasks == []


def test_plan_model_dump():
    """Plan supports model_dump for serialization."""
    p = Plan(tasks=[TaskModel(id="1", description="test")], rationale="ok")
    dump = p.model_dump()
    assert dump["rationale"] == "ok"
    assert dump["tasks"][0]["description"] == "test"


# --- TestResult ---

def test_test_result_defaults():
    """TestResult has correct defaults."""
    t = TestResult(passed=True, summary="ok")
    assert t.passed is True
    assert t.summary == "ok"
    assert t.failures == []


def test_test_result_with_failures():
    """TestResult stores failure messages."""
    t = TestResult(passed=False, summary="2 tests failed", failures=["err1", "err2"])
    assert t.passed is False
    assert len(t.failures) == 2
    assert t.failures[0] == "err1"
