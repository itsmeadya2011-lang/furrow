"""Tests for PlannerAgent and TesterAgent."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.config import TaskModel


def _make_client() -> MagicMock:
    client = MagicMock()
    client.settings = MagicMock()
    client.settings.planner_model = "test-planner"
    client.settings.tester_model = "test-tester"
    client.settings.max_tokens = 4096
    client.settings.timeout = 30.0
    client.settings.max_retries = 1
    client.settings.retry_delay = 0.01
    client.complete = AsyncMock(return_value="")
    return client


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_plan_parses_valid_json(self):
        client = _make_client()
        plan_json = json.dumps({
            "tasks": [
                {"id": "1", "description": "Add auth", "files": ["auth.py"], "dependencies": []}
            ],
            "rationale": "Need auth"
        })
        client.complete.return_value = plan_json

        planner = PlannerAgent(client=client)
        plan = await planner.plan("Add JWT auth")

        assert len(plan.tasks) == 1
        assert plan.tasks[0].description == "Add auth"
        assert plan.rationale == "Need auth"
        assert client.complete.called

    @pytest.mark.asyncio
    async def test_plan_passes_system_prompt(self):
        client = _make_client()
        plan_json = json.dumps({"tasks": [], "rationale": "done"})
        client.complete.return_value = plan_json

        planner = PlannerAgent(client=client)
        await planner.plan("some goal")

        call_kwargs = client.complete.call_args
        # system should be passed as a keyword argument
        assert "system" in call_kwargs.kwargs
        system_arg = call_kwargs.kwargs["system"]
        assert "planning agent" in system_arg.lower()

    @pytest.mark.asyncio
    async def test_plan_invalid_json_raises(self):
        client = _make_client()
        client.complete.return_value = "this is not json"

        planner = PlannerAgent(client=client)
        with pytest.raises(ValueError, match="Failed to parse plan"):
            await planner.plan("some goal")

    @pytest.mark.asyncio
    async def test_plan_empty_tasks(self):
        client = _make_client()
        client.complete.return_value = json.dumps({"tasks": [], "rationale": "already done"})

        planner = PlannerAgent(client=client)
        plan = await planner.plan("finish the project")
        assert plan.tasks == []
        assert plan.rationale == "already done"


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_run_tests_passes(self):
        client = _make_client()
        client.complete.return_value = json.dumps({
            "passed": True,
            "summary": "All 42 tests passed",
            "failures": []
        })

        tester = TesterAgent(client=client)
        # Mock _run_tests to avoid spawning real subprocesses
        tester._run_tests = AsyncMock(return_value="42 passed in 0.5s")

        task = TaskModel(id="1", description="done")
        result = await tester.run("the goal", [task])

        assert result.passed is True
        assert result.summary == "All 42 tests passed"
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_run_tests_fails(self):
        client = _make_client()
        client.complete.return_value = json.dumps({
            "passed": False,
            "summary": "1 test failed",
            "failures": ["test_auth failed: assertion error"]
        })

        tester = TesterAgent(client=client)
        tester._run_tests = AsyncMock(return_value="FAILED tests/test_auth.py")

        task = TaskModel(id="1", description="done")
        result = await tester.run("the goal", [task])

        assert result.passed is False
        assert "test_auth" in result.failures[0]

    @pytest.mark.asyncio
    async def test_run_tests_fallback_on_non_json(self):
        client = _make_client()
        client.complete.return_value = "The tests all passed successfully"

        tester = TesterAgent(client=client)
        tester._run_tests = AsyncMock(return_value="all green")

        task = TaskModel(id="1", description="done")
        result = await tester.run("the goal", [task])

        assert result.passed is True
        assert result.summary == "The tests all passed successfully"

    @pytest.mark.asyncio
    async def test_run_tests_fallback_non_json_failed(self):
        client = _make_client()
        client.complete.return_value = "tests failed badly"

        tester = TesterAgent(client=client)
        tester._run_tests = AsyncMock(return_value="error output")

        task = TaskModel(id="1", description="done")
        result = await tester.run("the goal", [task])

        assert result.passed is False

    @pytest.mark.asyncio
    async def test_test_runner_not_found(self):
        """When no test runner is found, _run_tests returns a message string."""
        client = _make_client()
        tester = TesterAgent(client=client)
        # Mock create_subprocess_exec to simulate all commands not being found
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(side_effect=FileNotFoundError)):
            test_output = await tester._run_tests()

        assert isinstance(test_output, str)
        assert "No test runner found" in test_output
