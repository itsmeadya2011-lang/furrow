from pathlib import Path

import pytest

from furrow.config import (
    Plan,
    Provider,
    Settings,
    TaskModel,
    TestResult,
)


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_provider_values():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.model == "claude-sonnet-4-20250514"
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.workspace == Path.cwd()


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_plan_and_result_construction():
    p = Plan(
        tasks=[TaskModel(id="1", description="write code", files=["a.py"])],
        rationale="needed",
    )
    assert len(p.tasks) == 1
    assert p.tasks[0].files == ["a.py"]
    assert p.rationale == "needed"

    r = TestResult(passed=False, summary="failed", failures=["assert x"])
    assert r.passed is False
    assert r.failures == ["assert x"]
