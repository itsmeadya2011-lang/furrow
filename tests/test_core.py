import json
from unittest.mock import AsyncMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_is_done_no_plan(self):
        orch = Orchestrator(goal="test")
        assert orch._is_done() is True

    @pytest.mark.asyncio
    async def test_is_done_with_pending_tasks(self):
        orch = Orchestrator(goal="test")
        orch.plan = Plan(
            tasks=[TaskModel(id="1", description="a"), TaskModel(id="2", description="b")],
            rationale="ok",
        )
        assert orch._is_done() is False

    @pytest.mark.asyncio
    async def test_is_done_with_completed_tasks(self):
        orch = Orchestrator(goal="test")
        orch.plan = Plan(
            tasks=[TaskModel(id="1", description="a", status="completed")],
            rationale="ok",
        )
        assert orch._is_done() is True

    @pytest.mark.asyncio
    async def test_is_done_with_failed_tasks(self):
        orch = Orchestrator(goal="test")
        orch.plan = Plan(
            tasks=[TaskModel(id="1", description="a", status="failed")],
            rationale="ok",
        )
        assert orch._is_done() is False

    @pytest.mark.asyncio
    async def test_get_tasks_returns_plan_tasks(self):
        orch = Orchestrator(goal="test")
        orch.plan = Plan(
            tasks=[TaskModel(id="1", description="a")],
            rationale="ok",
        )
        assert len(orch._get_tasks()) == 1
        assert orch._get_tasks()[0].id == "1"


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_plan_parses_json(self):
        mock_client = AsyncMock()
        mock_client.complete.return_value = json.dumps({
            "tasks": [
                {"id": "1", "description": "task one", "files": [], "dependencies": []}
            ],
            "rationale": "plan",
        })

        agent = PlannerAgent(client=mock_client)
        plan = await agent.plan("do something")
        assert len(plan.tasks) == 1
        assert plan.tasks[0].description == "task one"

    @pytest.mark.asyncio
    async def test_plan_raises_on_invalid_json(self):
        mock_client = AsyncMock()
        mock_client.complete.return_value = "not json"

        agent = PlannerAgent(client=mock_client)
        with pytest.raises(ValueError, match="Failed to parse plan"):
            await agent.plan("do something")


class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_run_returns_completion(self):
        mock_client = AsyncMock()
        mock_client.complete.return_value = "Implemented feature X"

        task = TaskModel(id="1", description="implement X")
        agent = WorkerAgent(task=task, client=mock_client)
        result = await agent.run()
        assert result == "Implemented feature X"


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_run_passed_tests(self):
        mock_client = AsyncMock()
        mock_client.complete.return_value = json.dumps({
            "passed": True,
            "summary": "All tests passed",
            "failures": [],
        })

        agent = TesterAgent(client=mock_client)
        result = await agent.run("goal", [])
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_run_failed_tests_heuristic(self):
        mock_client = AsyncMock()
        mock_client.complete.return_value = "Something went wrong but no json"

        agent = TesterAgent(client=mock_client)
        result = await agent.run("goal", [])
        assert result.passed is False


class TestLLMClient:
    def test_anthropic_initialization(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            client = LLMClient()
            assert client.anthropic is not None

    def test_openai_initialization(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = LLMClient()
            assert client.openai is not None

    def test_anthropic_missing_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            client = LLMClient()
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
                _ = client.anthropic

    def test_openai_missing_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            client = LLMClient()
            with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
                _ = client.openai
