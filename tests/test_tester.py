from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import TaskModel, TestResult


class TestTesterAgent:
    def test_tester_no_runner_returns_failure(self, tmp_path: Path) -> None:
        """When no test runner is found, _run_tests() returns 'No test runner found.'"""
        mock_client = MagicMock()
        mock_client.settings.workspace = tmp_path
        agent = TesterAgent(client=mock_client)

        # Patch create_subprocess_exec to always raise FileNotFoundError
        with patch(
            "furrow.agents.tester.asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("not found"),
        ):
            result = asyncio.run(agent._run_tests())

        assert result == "No test runner found."

    @pytest.mark.asyncio
    async def test_tester_parses_json_result(self) -> None:
        """When LLM returns valid JSON, TestResult is parsed correctly."""
        mock_client = AsyncMock()
        mock_client.settings.tester_model = "claude-3-5-sonnet-20241022"
        mock_client.complete.return_value = '{"passed": true, "summary": "All tests passed", "failures": []}'

        agent = TesterAgent(client=mock_client)

        # Mock _run_tests to return some output
        agent._run_tests = AsyncMock(return_value="test output here")

        result = await agent.run("test goal", [])

        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.summary == "All tests passed"
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_tester_handles_non_json_response(self) -> None:
        """When LLM returns non-JSON, falls back to text-based parsing."""
        mock_client = AsyncMock()
        mock_client.settings.tester_model = "claude-3-5-sonnet-20241022"
        mock_client.complete.return_value = "The tests passed successfully!"

        agent = TesterAgent(client=mock_client)

        # Mock _run_tests to return some output
        agent._run_tests = AsyncMock(return_value="test output here")

        result = await agent.run("test goal", [])

        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.summary == "The tests passed successfully!"
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_tester_handles_non_json_response_failed(self) -> None:
        """When LLM returns non-JSON with 'failed' in text, passed should be False."""
        mock_client = AsyncMock()
        mock_client.settings.tester_model = "claude-3-5-sonnet-20241022"
        mock_client.complete.return_value = "The tests failed with errors."

        agent = TesterAgent(client=mock_client)

        # Mock _run_tests to return some output
        agent._run_tests = AsyncMock(return_value="test output here")

        result = await agent.run("test goal", [])

        assert isinstance(result, TestResult)
        assert result.passed is False
        assert result.summary == "The tests failed with errors."
        assert result.failures == []
