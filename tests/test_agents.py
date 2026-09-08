import json
from unittest.mock import AsyncMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult


class MockClient:
    def __init__(self, responses=None):
        self.responses = responses or []
        self.settings = type("Settings", (), {"planner_model": "p", "worker_model": "w", "tester_model": "t"})()

    async def complete(self, prompt, model=None, **kwargs):
        return self.responses.pop(0)


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_valid_json_plan(self):
        client = MockClient(responses=[json.dumps({
            "tasks": [{"id": "1", "description": "task"}],
            "rationale": "ok",
        })])
        agent = PlannerAgent(client=client)
        plan = await agent.plan("do thing")
        assert isinstance(plan, Plan)
        assert plan.tasks[0].description == "task"

    @pytest.mark.asyncio
    async def test_fallback_on_bad_json(self):
        client = MockClient(responses=["not json", '{"tasks": [], "rationale": "ok"}'])
        agent = PlannerAgent(client=client)
        plan = await agent.plan("do thing")
        assert isinstance(plan, Plan)
        assert plan.rationale == "ok"

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        client = MockClient(responses=["bad", "bad", "bad"])
        agent = PlannerAgent(client=client)
        plan = await agent.plan("do thing")
        assert plan.tasks == []
        assert "Failed" in plan.rationale


class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_worker_runs(self):
        client = MockClient(responses=["result"])
        agent = WorkerAgent(task=TaskModel(id="1", description="do work"), client=client)
        result = await agent.run()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_worker_retry(self):
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("transient error")
            return "success"

        mock_client = type("Client", (), {
            "settings": type("Settings", (), {"worker_model": "w"})(),
            "complete": side_effect,
        })()
        agent = WorkerAgent(task=TaskModel(id="1", description="do work"), client=mock_client)
        result = await agent.run()
        assert result == "success"
        assert call_count == 3


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_valid_json_parsing(self):
        client = MockClient(responses=[json.dumps({
            "passed": True,
            "summary": "all good",
            "failures": [],
        })])
        agent = TesterAgent(client=client)

        with patch.object(agent, "_run_tests", new_callable=AsyncMock, return_value="tests passed"):
            result = await agent.run("goal", [])

        assert result.passed is True
        assert result.summary == "all good"

    @pytest.mark.asyncio
    async def test_json_parsing_fallback(self):
        client = MockClient(responses=["tests passed"])
        agent = TesterAgent(client=client)

        with patch.object(agent, "_run_tests", new_callable=AsyncMock, return_value="tests passed"):
            result = await agent.run("goal", [])

        assert "passed" in result.summary.lower()

    @pytest.mark.asyncio
    async def test_test_run_failure(self):
        client = MockClient(responses=[])
        agent = TesterAgent(client=client)

        with patch.object(agent, "_run_tests", new_callable=AsyncMock, side_effect=Exception("test crashed")):
            result = await agent.run("goal", [])

        assert result.passed is False
        assert "test crashed" in result.summary
