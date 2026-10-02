"""Tests for the TesterAgent."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import TaskModel, TestResult


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.settings.tester_model = "claude-3-5-sonnet-20241022"
    client.complete = AsyncMock()
    return client


class TestTesterRun:
    @pytest.mark.asyncio
    async def test_valid_test_result_passed(self, mock_client):
        mock_client.complete.return_value = json.dumps({
            "passed": True,
            "summary": "All 5 tests passed",
            "failures": [],
        })

        agent = TesterAgent(client=mock_client)
        agent._run_tests = AsyncMock(return_value="")
        result = await agent.run("build API", [TaskModel(id="1", description="task")])

        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.summary == "All 5 tests passed"
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_valid_test_result_failed(self, mock_client):
        mock_client.complete.return_value = json.dumps({
            "passed": False,
            "summary": "2 tests failed",
            "failures": ["test_auth.py::TestAuth::test_login failed", "test_auth.py::TestAuth::test_logout failed"],
        })

        agent = TesterAgent(client=mock_client)
        agent._run_tests = AsyncMock(return_value="")
        result = await agent.run("build API", [TaskModel(id="1", description="task")])

        assert result.passed is False
        assert len(result.failures) == 2
        assert "test_login" in result.failures[0]

    @pytest.mark.asyncio
    async def test_fallback_when_json_invalid(self, mock_client):
        """If LLM returns non-JSON, fall back to keyword detection."""
        mock_client.complete.return_value = "Tests passed successfully, all good"

        agent = TesterAgent(client=mock_client)
        agent._run_tests = AsyncMock(return_value="")
        result = await agent.run("build API", [])

        assert result.passed is True
        assert "all good" in result.summary.lower()

    @pytest.mark.asyncio
    async def test_fallback_when_json_indicates_failure(self, mock_client):
        mock_client.complete.return_value = "Tests failed, there are issues"

        agent = TesterAgent(client=mock_client)
        agent._run_tests = AsyncMock(return_value="")
        result = await agent.run("build API", [])

        assert result.passed is False

    @pytest.mark.asyncio
    async def test_test_execution_error(self, mock_client):
        """If _run_tests raises, return a failed TestResult."""
        agent = TesterAgent(client=mock_client)
        # Mock _run_tests to raise
        agent._run_tests = AsyncMock(side_effect=RuntimeError("pytest not found"))

        result = await agent.run("build API", [TaskModel(id="1", description="task")])

        assert result.passed is False
        assert "pytest not found" in result.summary
        assert len(result.failures) == 1

    @pytest.mark.asyncio
    async def test_prompt_includes_goal_and_test_output(self, mock_client):
        mock_client.complete.return_value = json.dumps({
            "passed": True, "summary": "ok", "failures": []
        })

        agent = TesterAgent(client=mock_client)
        agent._run_tests = AsyncMock(return_value="")
        await agent.run("Build a calculator", [TaskModel(id="1", description="calc")])

        args, kwargs = mock_client.complete.call_args
        prompt = args[0] if args else kwargs.get("prompt", "")
        assert "Build a calculator" in prompt
