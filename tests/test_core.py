"""Tests for core orchestrator, LLM client, and config models."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from openai import AsyncOpenAI

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.agents.prompts import (
    PLANNER_USER_TEMPLATE,
    TESTER_USER_TEMPLATE,
    WORKER_USER_TEMPLATE,
)
from furrow.llm import LLMClient


class _StubLLM:
    def __init__(self, settings):
        self.settings = settings


# --- Existing tests ---


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# --- Settings & model defaults ---


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0


def test_task_model_status_default():
    assert TaskModel(id="x", description="y").status == "pending"


def test_plan_with_empty_tasks():
    p = Plan(tasks=[], rationale="done")
    assert p.tasks == []


def test_test_result_default_failures():
    assert TestResult(passed=False, summary="x").failures == []


# --- Orchestrator state & lifecycle ---


def test_orchestrator_init_preserves_goal():
    stub = _StubLLM(settings=Settings())
    orch = Orchestrator(goal="build X", client=stub)
    assert orch.goal == "build X"
    assert orch.original_goal == "build X"
    assert orch.history == []
    assert orch.cycles == 0


def test_orchestrator_is_done_max_cycles():
    stub = _StubLLM(settings=Settings(max_cycles=3))
    orch = Orchestrator(goal="x", client=stub)
    orch.cycles = 5
    assert orch._is_done() is True


def test_orchestrator_is_done_no_tasks():
    stub = _StubLLM(settings=Settings())
    orch = Orchestrator(goal="x", client=stub)
    assert orch._is_done() is True


def test_orchestrator_is_done_with_pending_task():
    stub = _StubLLM(settings=Settings())
    orch = Orchestrator(goal="x", client=stub)
    orch.history.append(TaskModel(id="1", description="do", status="pending"))
    assert orch._is_done() is False


def test_orchestrator_is_done_all_completed():
    stub = _StubLLM(settings=Settings())
    orch = Orchestrator(goal="x", client=stub)
    orch.history.append(TaskModel(id="1", description="a", status="completed"))
    orch.history.append(TaskModel(id="2", description="b", status="completed"))
    assert orch._is_done() is True


# --- History merging ---


def test_orchestrator_merge_history():
    stub = _StubLLM(settings=Settings())
    orch = Orchestrator(goal="x", client=stub)

    plan_a = Plan(tasks=[TaskModel(id="1", description="a")], rationale="r")
    orch._merge_history(plan_a.tasks)
    assert len(orch.history) == 1
    assert orch.history[0].id == "1"

    plan_b = Plan(tasks=[TaskModel(id="2", description="b")], rationale="r")
    orch._merge_history(plan_b.tasks)
    assert len(orch.history) == 2
    assert {t.id for t in orch.history} == {"1", "2"}


def test_orchestrator_merge_history_replace():
    stub = _StubLLM(settings=Settings())
    orch = Orchestrator(goal="x", client=stub)

    plan_old = Plan(tasks=[TaskModel(id="1", description="old")], rationale="r")
    orch._merge_history(plan_old.tasks)

    plan_new = Plan(tasks=[TaskModel(id="1", description="new")], rationale="r")
    orch._merge_history(plan_new.tasks)

    assert len(orch.history) == 1
    assert orch.history[0].id == "1"
    assert orch.history[0].description == "new"


# --- Concurrency / cycle execution ---


async def test_orchestrator_concurrency_limiter():
    stub = _StubLLM(settings=Settings(max_parallel_tasks=2))
    orch = Orchestrator(goal="noop", client=stub)

    mock_planner = AsyncMock()
    mock_planner.plan.return_value = Plan(tasks=[], rationale="none")
    orch.planner = mock_planner

    await orch._cycle()


# --- LLM client provider dispatch ---


def test_llm_client_provider_dispatch():
    anthropic_client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC))
    assert hasattr(anthropic_client, "_anthropic")

    openai_client = LLMClient(settings=Settings(provider=Provider.OPENAI))
    assert hasattr(openai_client, "_openai")

    ollama_client = LLMClient(settings=Settings(provider=Provider.OLLAMA))
    assert hasattr(ollama_client, "_ollama")


def test_llm_client_ollama_property_creates_client():
    client = LLMClient(settings=Settings(provider=Provider.OLLAMA))
    ollama = client.ollama
    assert isinstance(ollama, AsyncOpenAI)
    assert str(ollama.base_url).endswith("/v1")


# --- Prompt template formatting ---


def test_planner_user_template_format():
    out = PLANNER_USER_TEMPLATE.format(goal="g", state="(none)")
    assert isinstance(out, str)
    assert len(out) > 0
    assert "g" in out


def test_worker_user_template_format():
    out = WORKER_USER_TEMPLATE.format(description="d", files="any")
    assert isinstance(out, str)
    assert len(out) > 0
    assert "d" in out


def test_tester_user_template_format():
    out = TESTER_USER_TEMPLATE.format(goal="g", tasks="t", test_output="o")
    assert isinstance(out, str)
    assert len(out) > 0
    assert "g" in out
