"""Comprehensive tests for Furrow covering config, plan/test models, orchestrator logic,
LLM client dispatch, and the planner/tester/worker agents.

All LLM-driven behavior is exercised through mocks; no real network calls are made.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from openai import AsyncOpenAI
from pydantic import ValidationError

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_provider_enum_members():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"
    assert {Provider.ANTHROPIC, Provider.OPENAI, Provider.OLLAMA} == {
        Provider.ANTHROPIC,
        Provider.OPENAI,
        Provider.OLLAMA,
    }
    assert len(list(Provider)) == 3


def test_settings_defaults(monkeypatch):
    for key in list(os.environ.keys()):
        if key.startswith("FURROW_"):
            monkeypatch.delenv(key)
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.ollama_model == "llama3.1"
    assert s.timeout == 120.0
    assert s.verbose is False


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.status == "pending"
    assert t.files == []
    assert t.dependencies == []
    assert t.result is None


def test_task_model_accepts_overrides():
    t = TaskModel(
        id="2",
        description="do other",
        files=["a.py"],
        dependencies=["1"],
        status="completed",
        result="done",
    )
    assert t.files == ["a.py"]
    assert t.dependencies == ["1"]
    assert t.status == "completed"
    assert t.result == "done"


# ---------------------------------------------------------------------------
# Plan / TestResult tests
# ---------------------------------------------------------------------------


def test_plan_with_multiple_tasks_and_rationale():
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task one"),
            TaskModel(id="2", description="task two", files=["b.py"]),
        ],
        rationale="break down the work",
    )
    assert len(plan.tasks) == 2
    assert plan.rationale == "break down the work"
    assert plan.tasks[0].id == "1"
    assert plan.tasks[1].description == "task two"
    assert plan.tasks[1].files == ["b.py"]


def test_plan_requires_fields():
    with pytest.raises(ValidationError):
        Plan(rationale="missing tasks")


def test_test_result_has_test_attribute_false():
    assert TestResult.__test__ is False
    assert "__test__" not in TestResult.model_fields


def test_test_result_defaults():
    t = TestResult(passed=True, summary="ok")
    assert t.passed is True
    assert t.summary == "ok"
    assert t.failures == []


def test_test_result_is_not_collected_by_pytest():
    # pytest must not treat this pydantic model as a test class.
    assert TestResult.__test__ is False


# ---------------------------------------------------------------------------
# Orchestrator tests (mocked LLM, no real calls)
# ---------------------------------------------------------------------------


def _make_orch() -> Orchestrator:
    return Orchestrator(goal="a goal", client=MagicMock())


def test_orchestrator_get_tasks_returns_plan_tasks_when_set():
    orch = _make_orch()
    plan = Plan(tasks=[TaskModel(id="1", description="x")], rationale="r")
    orch.plan = plan
    assert orch._get_tasks() is plan.tasks


def test_orchestrator_get_tasks_returns_empty_when_plan_none():
    orch = _make_orch()
    orch.plan = None
    assert orch._get_tasks() == []


def test_orchestrator_is_done_true_when_all_completed():
    orch = _make_orch()
    orch.plan = Plan(
        tasks=[
            TaskModel(id="1", description="x", status="completed"),
            TaskModel(id="2", description="y", status="completed"),
        ],
        rationale="r",
    )
    assert orch._is_done() is True


def test_orchestrator_is_done_false_when_any_failed():
    orch = _make_orch()
    orch.plan = Plan(
        tasks=[
            TaskModel(id="1", description="x", status="completed"),
            TaskModel(id="2", description="y", status="failed"),
        ],
        rationale="r",
    )
    assert orch._is_done() is False


def test_orchestrator_is_done_false_when_some_pending():
    orch = _make_orch()
    orch.plan = Plan(
        tasks=[
            TaskModel(id="1", description="x", status="completed"),
            TaskModel(id="2", description="y", status="pending"),
        ],
        rationale="r",
    )
    assert orch._is_done() is False


def test_orchestrator_is_done_true_when_zero_tasks():
    orch = _make_orch()
    orch.plan = Plan(tasks=[], rationale="r")
    assert orch._is_done() is True


def test_orchestrator_is_done_true_when_plan_is_none():
    orch = _make_orch()
    orch.plan = None
    # No plan -> no tasks -> vacuously done.
    assert orch._is_done() is True


async def test_orchestrator_max_cycles_enforcement():
    max_cycles = 2
    with patch("furrow.core.orchestrator.settings") as mock_settings:
        mock_settings.max_cycles = max_cycles
        mock_settings.max_parallel_tasks = 5
        orch = Orchestrator(goal="max cycles goal", client=MagicMock())

        plan = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
        orch.planner = MagicMock()
        orch.planner.plan = AsyncMock(return_value=plan)
        # Worker marks the task completed each cycle.
        orch._run_worker = AsyncMock(return_value="completed work")
        # Force not-done so the loop only stops due to max_cycles.
        orch._is_done = Mock(return_value=False)

        with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
            MockTester.return_value.run = AsyncMock(
                return_value=TestResult(passed=False, summary="fail", failures=["x"])
            )
            await orch.run()

        # The loop should run exactly max_cycles full cycles before breaking.
        assert orch.planner.plan.call_count == max_cycles
        # Tasks were marked completed each cycle.
        assert orch.plan.tasks[0].status == "completed"
        assert orch.cycles == max_cycles + 1


async def test_orchestrator_run_stops_when_done():
    """When _is_done returns True after the first cycle, no max_cycles needed."""
    with patch("furrow.core.orchestrator.settings") as mock_settings:
        mock_settings.max_cycles = 0
        mock_settings.max_parallel_tasks = 5
        orch = Orchestrator(goal="done goal", client=MagicMock())

        plan = Plan(
            tasks=[TaskModel(id="1", description="do thing", status="pending")], rationale="ok"
        )
        orch.planner = MagicMock()
        orch.planner.plan = AsyncMock(return_value=plan)
        orch._run_worker = AsyncMock(return_value="done")

        with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
            MockTester.return_value.run = AsyncMock(
                return_value=TestResult(passed=True, summary="ok", failures=[])
            )
            await orch.run()

        assert orch._is_done() is True
        assert orch.planner.plan.call_count == 1
        assert orch.plan.tasks[0].status == "completed"


# ---------------------------------------------------------------------------
# LLM Client tests (mocked providers, no network)
# ---------------------------------------------------------------------------


def _client(provider: Provider = Provider.OLLAMA) -> LLMClient:
    return LLMClient(settings=Settings(provider=provider))


async def test_complete_dispatches_to_anthropic():
    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC))
    client._complete_anthropic = AsyncMock(return_value="anthropic")
    client._complete_openai = AsyncMock(return_value="openai")
    client._complete_ollama = AsyncMock(return_value="ollama")
    result = await client.complete("prompt")
    assert result == "anthropic"
    client._complete_anthropic.assert_awaited_once()
    client._complete_openai.assert_not_awaited()
    client._complete_ollama.assert_not_awaited()


async def test_complete_dispatches_to_openai():
    client = LLMClient(settings=Settings(provider=Provider.OPENAI))
    client._complete_anthropic = AsyncMock(return_value="anthropic")
    client._complete_openai = AsyncMock(return_value="openai")
    client._complete_ollama = AsyncMock(return_value="ollama")
    result = await client.complete("prompt")
    assert result == "openai"
    client._complete_openai.assert_awaited_once()
    client._complete_anthropic.assert_not_awaited()
    client._complete_ollama.assert_not_awaited()


async def test_complete_dispatches_to_ollama():
    client = LLMClient(settings=Settings(provider=Provider.OLLAMA))
    client._complete_anthropic = AsyncMock(return_value="anthropic")
    client._complete_openai = AsyncMock(return_value="openai")
    client._complete_ollama = AsyncMock(return_value="ollama")
    result = await client.complete("prompt")
    assert result == "ollama"
    client._complete_ollama.assert_awaited_once()
    client._complete_anthropic.assert_not_awaited()
    client._complete_openai.assert_not_awaited()


async def test_complete_ollama_calls_ollama_chat_completions_create():
    client = LLMClient(settings=Settings(provider=Provider.OLLAMA))
    fake_response = MagicMock()
    fake_response.choices[0].message.content = "ollama result"
    client._ollama = MagicMock()
    client._ollama.chat.completions.create = AsyncMock(return_value=fake_response)
    result = await client._complete_ollama("prompt", "", "model")
    assert result == "ollama result"
    client._ollama.chat.completions.create.assert_awaited_once()
    # Verify the kwargs passed to the upstream call.
    call_kwargs = client._ollama.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "model"
    assert call_kwargs["messages"][1]["role"] == "user"
    assert "prompt" in call_kwargs["messages"][1]["content"]


def test_ollama_property_creates_async_openai_with_base_url():
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://test:11434")
    client = LLMClient(settings=s)
    c = client.ollama
    assert isinstance(c, AsyncOpenAI)
    assert str(c.base_url) == "http://test:11434"


# ---------------------------------------------------------------------------
# PlannerAgent tests (mocked LLMClient.complete)
# ---------------------------------------------------------------------------


def _planner_client(settings: Settings | None = None) -> LLMClient:
    return LLMClient(settings=settings or Settings(provider=Provider.OLLAMA))


def _valid_plan_json() -> str:
    return (
        '{"tasks": [{"id": "1", "description": "do thing", "files": [], "dependencies": []}], '
        '"rationale": "ok"}'
    )


async def test_planner_plan_parses_valid_json_into_plan():
    client = _planner_client()
    agent = PlannerAgent(client=client)
    client.complete = AsyncMock(return_value=_valid_plan_json())
    plan = await agent.plan("build something")
    assert isinstance(plan, Plan)
    assert plan.rationale == "ok"
    assert plan.tasks[0].description == "do thing"
    client.complete.assert_awaited_once()


async def test_planner_plan_retries_on_invalid_json_then_succeeds():
    client = _planner_client()
    agent = PlannerAgent(client=client)
    client.complete = AsyncMock(side_effect=["{not valid json", _valid_plan_json()])
    plan = await agent.plan("build something")
    assert isinstance(plan, Plan)
    assert plan.tasks[0].description == "do thing"
    assert client.complete.await_count == 2


async def test_planner_plan_raises_value_error_when_both_attempts_invalid():
    client = _planner_client()
    agent = PlannerAgent(client=client)
    client.complete = AsyncMock(return_value="<<<not json at all>>>")
    with pytest.raises(ValueError):
        await agent.plan("build something")
    assert client.complete.await_count == 2


# ---------------------------------------------------------------------------
# TesterAgent tests (mocked LLMClient.complete / subprocess)
# ---------------------------------------------------------------------------


async def test_tester_run_tests_no_runner_found():
    client = LLMClient(settings=Settings(provider=Provider.OLLAMA))
    agent = TesterAgent(client=client)
    with patch(
        "furrow.agents.tester.asyncio.create_subprocess_exec",
        side_effect=FileNotFoundError("no command"),
    ):
        result = await agent._run_tests()
    assert result == "No test runner found."


async def test_tester_run_returns_passed_true_from_valid_json():
    client = LLMClient(settings=Settings(provider=Provider.OLLAMA))
    agent = TesterAgent(client=client)
    agent._run_tests = AsyncMock(return_value="test output here")
    response = '{"passed": true, "summary": "All tests passed", "failures": []}'
    client.complete = AsyncMock(return_value=response)
    result = await agent.run("a goal", [])
    assert isinstance(result, TestResult)
    assert result.passed is True
    assert result.summary == "All tests passed"
    assert result.failures == []


async def test_tester_run_returns_passed_true_via_fallback_text():
    client = LLMClient(settings=Settings(provider=Provider.OLLAMA))
    agent = TesterAgent(client=client)
    agent._run_tests = AsyncMock(return_value="test output here")
    # Invalid JSON but contains "passed" -> fallback parsing.
    client.complete = AsyncMock(return_value="All tests passed successfully")
    result = await agent.run("a goal", [])
    assert isinstance(result, TestResult)
    assert result.passed is True


# ---------------------------------------------------------------------------
# WorkerAgent tests (mocked LLMClient.complete)
# ---------------------------------------------------------------------------


async def test_worker_run_constructs_prompt_with_description_and_files():
    client = LLMClient(settings=Settings(provider=Provider.OLLAMA))
    task = TaskModel(id="1", description="implement login flow", files=["src/auth.py"])
    agent = WorkerAgent(task=task, client=client)
    client.complete = AsyncMock(return_value="implemented")

    result = await agent.run()

    assert result == "implemented"
    assert client.complete.await_count == 1
    sent_prompt = client.complete.call_args.args[0]
    assert "implement login flow" in sent_prompt
    assert "src/auth.py" in sent_prompt
