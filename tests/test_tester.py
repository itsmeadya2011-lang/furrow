"""Tests for TesterAgent — test result parsing and test runner detection."""
from __future__ import annotations

import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import TaskModel, TestResult

from tests.conftest import FakeLLMClient, fake_settings  # noqa: F401


class TestTesterParse:
    async def test_run_valid_json_response(self, fake_client):
        fake_client._response = json.dumps({
            "passed": True,
            "summary": "All tests passed",
            "failures": [],
        })

        tester = TesterAgent(client=fake_client)
        with patch.object(tester, "_run_tests", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "test output"
            result = await tester.run("goal", [TaskModel(id="1", description="t")])

        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.summary == "All tests passed"
        assert result.failures == []

    async def test_run_json_with_failures(self, fake_client):
        fake_client._response = json.dumps({
            "passed": False,
            "summary": "2 tests failed",
            "failures": ["test_a failed", "test_b failed"],
        })

        tester = TesterAgent(client=fake_client)
        with patch.object(tester, "_run_tests", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "failure output"
            result = await tester.run("goal", [])

        assert result.passed is False
        assert len(result.failures) == 2
        assert "test_a failed" in result.failures

    async def test_run_invalid_json_falls_back_to_string_match(self, fake_client):
        fake_client._response = "the tests passed and everything is good"

        tester = TesterAgent(client=fake_client)
        with patch.object(tester, "_run_tests", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "output"
            result = await tester.run("goal", [])

        assert result.passed is True
        assert "passed" in result.summary.lower()

    async def test_run_invalid_json_falls_back_failed(self, fake_client):
        fake_client._response = "tests failed due to errors"

        tester = TesterAgent(client=fake_client)
        with patch.object(tester, "_run_tests", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "output"
            result = await tester.run("goal", [])

        assert result.passed is False

    async def test_run_handles_test_execution_error(self, fake_client):
        tester = TesterAgent(client=fake_client)
        with patch.object(tester, "_run_tests", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = RuntimeError("test runner crashed")
            result = await tester.run("goal", [])

        assert result.passed is False
        assert "test runner crashed" in result.summary or "test runner crashed" in result.failures[0]

    async def test_run_includes_goal_and_output_in_prompt(self, fake_client):
        fake_client._response = json.dumps({"passed": True, "summary": "ok", "failures": []})

        tester = TesterAgent(client=fake_client)
        with patch.object(tester, "_run_tests", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "pytest output here"
            await tester.run("my goal", [])

        prompt = fake_client.complete_calls[0][0]
        assert "my goal" in prompt
        assert "pytest output here" in prompt


class TestTestRunnerDetection:
    async def test_no_test_runner_returns_message(self, fake_client):
        tester = TesterAgent(client=fake_client)
        with patch("furrow.agents.tester.asyncio.create_subprocess_exec") as mock_sub:
            mock_sub.side_effect = FileNotFoundError("pytest not found")
            result = await tester._run_tests()
        assert "No test runner found" in result

    async def test_timeout_continues_to_next_candidate(self, fake_client):
        tester = TesterAgent(client=fake_client)
        proc_mock = MagicMock()
        proc_mock.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        proc_mock.kill = MagicMock()

        with patch("furrow.agents.tester.asyncio.create_subprocess_exec") as mock_sub:
            mock_sub.return_value = proc_mock
            result = await tester._run_tests()
        assert "No test runner found" in result

    async def test_successful_test_run_returns_output(self, fake_client):
        tester = TesterAgent(client=fake_client)

        async def mock_create_subprocess(*args, **kwargs):
            proc = MagicMock()
            proc.communicate = AsyncMock(return_value=(b"10 passed", b""))
            proc.kill = MagicMock()
            return proc

        with patch("furrow.agents.tester.asyncio.create_subprocess_exec", new=mock_create_subprocess):
            result = await tester._run_tests()
        assert "10 passed" in result
