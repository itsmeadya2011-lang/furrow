"""Tests for agents: PlannerAgent, WorkerAgent, TesterAgent."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, TestResult


def _make_settings(**overrides):
    kwargs = dict(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        planner_model="claude-3-5-haiku-20241022",
        worker_model="claude-3-5-sonnet-20241022",
        tester_model="claude-3-5-sonnet-20241022",
        anthropic_api_key="sk-test",
        max_parallel_tasks=5,
        max_cycles=0,
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_plan_returns_valid_plan(self):
        settings = _make_settings()
        client = MagicMock()
        client.settings = settings
        plan_json = json.dumps({
            "tasks": [
                {"id": "1", "description": "Do A", "files": [], "dependencies": []},
                {"id": "2", "description": "Do B", "files": ["b.py"], "dependencies": ["1"]},
            ],
            "rationale": "Split into two steps",
        })
        client.complete = AsyncMock(return_value=plan_json)

        agent = PlannerAgent(client=client)
        plan = await agent.plan("Build feature")

        assert len(plan.tasks) == 2
        assert plan.tasks[0].id == "1"
        assert plan.tasks[1].dependencies == ["1"]
        assert plan.rationale == "Split into two steps"
        # Should have called complete with the planner model
        client.complete.assert_called_once()
        _, kwargs = client.complete.call_args
        assert kwargs.get("model") == settings.planner_model

    @pytest.mark.asyncio
    async def test_plan_raises_on_invalid_json(self):
        settings = _make_settings()
        client = MagicMock()
        client.settings = settings
        client.complete = AsyncMock(return_value="not json at all")

        agent = PlannerAgent(client=client)
        with pytest.raises(ValueError, match="Failed to parse plan"):
            await agent.plan("Build feature")


class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_worker_returns_result(self):
        settings = _make_settings()
        client = MagicMock()
        client.settings = settings
        client.complete = AsyncMock(return_value="Done! Changed file.py")

        task = TaskModel(id="1", description="Add feature", files=["file.py"])
        agent = WorkerAgent(task=task, client=client)
        result = await agent.run()

        assert result == "Done! Changed file.py"
        client.complete.assert_called_once()
        _, kwargs = client.complete.call_args
        assert kwargs.get("model") == settings.worker_model

    @pytest.mark.asyncio
    async def test_worker_empty_files_shows_any(self):
        settings = _make_settings()
        client = MagicMock()
        client.settings = settings
        client.complete = AsyncMock(return_value="result")

        task = TaskModel(id="1", description="Add feature")
        agent = WorkerAgent(task=task, client=client)
        await agent.run()

        args, _ = client.complete.call_args
        prompt = args[0] if args else ""
        assert "any" in prompt


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_tester_returns_passed_result(self):
        settings = _make_settings()
        client = MagicMock()
        client.settings = settings
        result_json = json.dumps({
            "passed": True,
            "summary": "All tests passed",
            "failures": [],
        })
        client.complete = AsyncMock(return_value=result_json)

        agent = TesterAgent(client=client)
        tasks = [TaskModel(id="1", description="task")]
        with patch("furrow.agents.tester.TesterAgent._run_tests", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "test output"
            result = await agent.run("goal", tasks)

        assert result.passed is True
        assert result.summary == "All tests passed"

    @pytest.mark.asyncio
    async def test_tester_returns_failed_result(self):
        settings = _make_settings()
        client = MagicMock()
        client.settings = settings
        result_json = json.dumps({
            "passed": False,
            "summary": "1 test failed",
            "failures": ["test_foo failed"],
        })
        client.complete = AsyncMock(return_value=result_json)

        agent = TesterAgent(client=client)
        tasks = [TaskModel(id="1", description="task")]
        with patch("furrow.agents.tester.TesterAgent._run_tests", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "FAIL test_foo"
            result = await agent.run("goal", tasks)

        assert result.passed is False
        assert "test_foo failed" in result.failures

    @pytest.mark.asyncio
    async def test_tester_fallback_on_invalid_json(self):
        settings = _make_settings()
        client = MagicMock()
        client.settings = settings
        client.complete = AsyncMock(return_value="tests passed successfully")

        agent = TesterAgent(client=client)
        tasks = [TaskModel(id="1", description="task")]
        with patch("furrow.agents.tester.TesterAgent._run_tests", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "all good"
            result = await agent.run("goal", tasks)

        assert result.passed is True
        assert result.summary == "tests passed successfully"

    @pytest.mark.asyncio
    async def test_tester_handles_no_test_runner(self):
        settings = _make_settings()
        client = MagicMock()
        client.settings = settings
        client.complete = AsyncMock(return_value="ok")

        agent = TesterAgent(client=client)
        tasks = [TaskModel(id="1", description="task")]
        with patch("furrow.agents.tester.TesterAgent._run_tests", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "No test runner found."
            result = await agent.run("goal", tasks)

        # Should still call the LLM to interpret results
        client.complete.assert_called_once()


def test_test_result_model():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True
    assert t.failures == []
