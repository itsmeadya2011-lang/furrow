from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import (
    NO_TEST_RUNNER_SENTINEL,
    TEST_RUNNER_TIMEOUT_SENTINEL,
    TesterAgent,
)
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, TestResult


async def test_tester_run_tests_no_runner_sentinel(monkeypatch):
    async def fake_exec(*args, **kwargs):
        raise FileNotFoundError("no runner")

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    agent = TesterAgent(client=AsyncMock(), settings=Settings())
    out = await agent._run_tests()
    assert out.startswith(NO_TEST_RUNNER_SENTINEL)


async def test_tester_run_returns_failed_for_no_runner(monkeypatch):
    async def fake_exec(*args, **kwargs):
        raise FileNotFoundError("no runner")

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    agent = TesterAgent(client=AsyncMock(), settings=Settings())
    result = await agent.run("goal", [])
    assert isinstance(result, TestResult)
    assert result.passed is False
    assert result.summary.startswith(NO_TEST_RUNNER_SENTINEL)


async def test_tester_run_returns_failed_when_run_tests_raises(monkeypatch):
    agent = TesterAgent(client=AsyncMock(), settings=Settings())

    async def fake(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(TesterAgent, "_run_tests", fake)
    result = await agent.run("goal", [])
    assert isinstance(result, TestResult)
    assert result.passed is False
    assert "boom" in result.summary


async def test_planner_raises_value_error_on_invalid_json():
    client = AsyncMock()
    client.complete = AsyncMock(return_value="not json")
    agent = PlannerAgent(client=client, settings=Settings())
    with pytest.raises(ValueError):
        await agent.plan("some goal")


async def test_worker_run_returns_client_string():
    client = AsyncMock()
    client.complete = AsyncMock(return_value="done")
    agent = WorkerAgent(task=TaskModel(id="1", description="x"), client=client, settings=Settings())
    out = await agent.run()
    assert out == "done"
    client.complete.assert_awaited_once()


def test_sentinels_are_strings():
    assert isinstance(NO_TEST_RUNNER_SENTINEL, str)
    assert isinstance(TEST_RUNNER_TIMEOUT_SENTINEL, str)
    assert NO_TEST_RUNNER_SENTINEL != TEST_RUNNER_TIMEOUT_SENTINEL
