from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.llm import ToolCall


def test_provider_values():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"


def test_provider_from_string():
    assert Provider("openai") is Provider.OPENAI
    assert Provider("anthropic") is Provider.ANTHROPIC


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert isinstance(s.workspace, Path)


def test_settings_max_cycles_override():
    s = Settings(max_cycles=10, max_parallel_tasks=2)
    assert s.max_cycles == 10
    assert s.max_parallel_tasks == 2


def test_settings_unknown_provider_raises():
    with pytest.raises(ValidationError):
        Settings(provider="bogus")  # type: ignore[arg-type]


def test_task_model_defaults():
    t = TaskModel(id="1", description="do x")
    assert t.status == "pending"
    assert t.result is None
    assert t.files == []
    assert t.dependencies == []


def test_plan_and_test_result():
    plan = Plan(tasks=[TaskModel(id="1", description="x")], rationale="r")
    assert plan.rationale == "r"
    tr = TestResult(passed=True, summary="ok")
    assert tr.passed is True
    assert tr.failures == []


def test_toolcall_dataclass():
    tc = ToolCall(id="1", name="read_file", args={"path": "x"})
    assert tc.id == "1"
    assert tc.name == "read_file"
    assert tc.args == {"path": "x"}
