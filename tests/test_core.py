from __future__ import annotations

from unittest import mock
from unittest.mock import AsyncMock

import pytest

import furrow.agents.tester as tester_module
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def _orch(goal="build a tool", **overrides):
    settings = Settings(**overrides)
    return Orchestrator(goal=goal, client=LLMClient(settings=settings))


def test_orchestrator_is_done_completed():
    orch = _orch()
    orch.cycles = 1
    orch.plan = Plan(tasks=[TaskModel(id="1", description="t", status="completed")], rationale="ok")
    orch.last_test_result = TestResult(passed=True, summary="ok", failures=[])
    assert orch._is_done() is True


def test_orchestrator_is_done_pending_not_done():
    orch = _orch()
    orch.cycles = 1
    orch.plan = Plan(tasks=[TaskModel(id="1", description="t", status="pending")], rationale="ok")
    orch.last_test_result = TestResult(passed=True, summary="ok", failures=[])
    assert orch._is_done() is False


def test_orchestrator_is_done_failed_not_done():
    orch = _orch()
    orch.cycles = 1
    orch.plan = Plan(tasks=[TaskModel(id="1", description="t", status="failed")], rationale="ok")
    orch.last_test_result = TestResult(passed=True, summary="ok", failures=[])
    assert orch._is_done() is False


def test_orchestrator_is_done_tests_failed_not_done():
    orch = _orch()
    orch.cycles = 1
    orch.plan = Plan(
        tasks=[TaskModel(id="1", description="t", status="completed")], rationale="ok"
    )
    orch.last_test_result = TestResult(passed=False, summary="fail", failures=["x"])
    assert orch._is_done() is False


def test_orchestrator_max_cycles_halt():
    orch = _orch(max_cycles=1)
    orch.cycles = 1
    orch.plan = Plan(
        tasks=[TaskModel(id="1", description="t", status="pending")], rationale="ok"
    )
    orch.last_test_result = None
    # pending task but max_cycles reached -> done
    assert orch._is_done() is True


def test_orchestrator_get_tasks_uses_plan():
    orch = _orch()
    orch.plan = Plan(tasks=[TaskModel(id="1", description="t")], rationale="ok")
    assert orch._get_tasks() == orch.plan.tasks
    orch.plan = None
    assert orch._get_tasks() == []


@pytest.mark.asyncio
async def test_tester_run_tests_no_runner():
    with mock.patch.object(
        tester_module.asyncio, "create_subprocess_exec", side_effect=FileNotFoundError
    ):
        tester = TesterAgent()
        assert await tester._run_tests() == "No test runner found."


@pytest.mark.asyncio
async def test_worker_run_uses_complete_with_tools():
    client = LLMClient()
    with mock.patch.object(
        client, "complete_with_tools", new_callable=AsyncMock
    ) as mocked:
        mocked.return_value = "Done: implemented feature X"
        worker = WorkerAgent(task=TaskModel(id="1", description="do X"), client=client)
        result = await worker.run()
    assert result == "Done: implemented feature X"
    mocked.assert_awaited_once()


@pytest.mark.asyncio
async def test_llm_retry_then_succeeds():
    import anthropic

    client = LLMClient(max_retries=3)
    call_count = 0

    async def fake_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise anthropic.APIError("transient", request=None, body=None)
        return mock.MagicMock(content=[mock.MagicMock(text="hello", type="text")])

    fake_anthropic = mock.MagicMock()
    fake_anthropic.messages.create = fake_create
    client._anthropic = fake_anthropic

    result = await client.complete("hi", system="s", model="m")
    assert result == "hello"
    assert call_count == 2


@pytest.mark.asyncio
async def test_llm_retry_exhausted_reraises():
    import anthropic

    client = LLMClient(max_retries=2)

    async def fake_create(**kwargs):
        raise anthropic.APIError("boom", request=None, body=None)

    fake_anthropic = mock.MagicMock()
    fake_anthropic.messages.create = fake_create
    client._anthropic = fake_anthropic

    with pytest.raises(anthropic.APIError):
        await client.complete("hi", system="s", model="m")
