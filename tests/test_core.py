import os
from pathlib import Path

import pytest
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_settings_defaults(monkeypatch):
    # Clear any env vars that could override config-model defaults.
    for key in list(os.environ):
        if key.startswith("FURROW_"):
            monkeypatch.delenv(key, raising=False)
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_cycles == 0
    assert s.max_parallel_tasks == 5
    assert isinstance(s.workspace, Path)


def test_provider_enum_values():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"


def test_task_model_defaults():
    t = TaskModel(id="1", description="x")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_plan_rationale_required():
    p = Plan(tasks=[], rationale="done")
    assert p.rationale == "done"
    # rationale has no default and is therefore required
    with pytest.raises(Exception):
        Plan(tasks=[])


def test_test_result_failures_default():
    t = TestResult(passed=False, summary="bad")
    assert t.failures == []
