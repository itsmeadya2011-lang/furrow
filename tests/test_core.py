from __future__ import annotations

import pytest

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_test_result_defaults():
    t = TestResult(passed=False, summary="broken")
    assert t.failures == []
    assert t.passed is False


def test_provider_values():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"


def test_task_model_defaults():
    t = TaskModel(id="a", description="d")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_plan_empty_tasks():
    p = Plan(tasks=[], rationale="nothing to do")
    assert p.tasks == []


def test_llm_client_requires_anthropic_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from furrow.llm import LLMClient

    s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)
    client = LLMClient(s)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        _ = client.anthropic
