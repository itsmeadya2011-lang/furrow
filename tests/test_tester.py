"""Tests for TesterAgent: command execution, test running, and result parsing."""
from __future__ import annotations

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import TaskModel, TestResult


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.settings = MagicMock()
    client.settings.tester_model = "sonnet"
    client.complete = AsyncMock(return_value='{"passed": true, "summary": "All tests passed", "failures": []}')
    return client


class TestRunCommand:
    @pytest.mark.asyncio
    async def test_returns_none_for_missing_command(self, mock_client):
        tester = TesterAgent(client=mock_client)
        result = await tester._run_command(["nonexistent_command_xyz", "--flag"])
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_output_for_valid_command(self, mock_client):
        tester = TesterAgent(client=mock_client)
        result = await tester._run_command([sys.executable, "--version"])
        assert result is not None
        assert "Python" in result

    @pytest.mark.asyncio
    async def test_returns_empty_string_for_silent_command(self, mock_client):
        tester = TesterAgent(client=mock_client)
        result = await tester._run_command(["true"])
        assert result is not None  # empty string, not None


class TestRunTests:
    @pytest.mark.asyncio
    async def test_no_test_runner_found(self, mock_client, tmp_path):
        tester = TesterAgent(client=mock_client)
        with patch.object(tester, "_run_command", AsyncMock(return_value=None)):
            result = await tester._run_tests()
        assert "No test runner found" in result


class TestRun:
    @pytest.mark.asyncio
    async def test_parses_valid_json_response(self, mock_client):
        tester = TesterAgent(client=mock_client)
        with patch.object(tester, "_run_tests", AsyncMock(return_value="test output")):
            result = await tester.run("test goal", [TaskModel(id="1", description="t")])
        assert result.passed is True
        assert result.summary == "All tests passed"

    @pytest.mark.asyncio
    async def test_handles_invalid_json_response(self, mock_client):
        mock_client.complete = AsyncMock(return_value="tests seem to have passed")
        tester = TesterAgent(client=mock_client)
        with patch.object(tester, "_run_tests", AsyncMock(return_value="test output")):
            result = await tester.run("test goal", [TaskModel(id="1", description="t")])
        assert result.passed is True
        assert "passed" in result.summary.lower()

    @pytest.mark.asyncio
    async def test_handles_failed_test_result(self, mock_client):
        mock_client.complete = AsyncMock(
            return_value=json.dumps({"passed": False, "summary": "1 failure", "failures": ["test_a"]})
        )
        tester = TesterAgent(client=mock_client)
        with patch.object(tester, "_run_tests", AsyncMock(return_value="FAIL")):
            result = await tester.run("test goal", [TaskModel(id="1", description="t")])
        assert result.passed is False
        assert "test_a" in result.failures
