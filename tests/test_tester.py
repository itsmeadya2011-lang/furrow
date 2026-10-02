"""Tests for the TesterAgent."""
import json

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import TestResult
from tests.conftest import MockLLMClient


class TestTesterTimeout:
    def test_timeout_constant(self):
        assert TesterAgent.TEST_TIMEOUT == 300


class TestTesterRun:
    @pytest.mark.asyncio
    async def test_no_test_runner_returns_testresult(self):
        client = MockLLMClient(
            responses=[json.dumps({
                "passed": True,
                "summary": "All tests passed (no runner found, but LLM says ok)",
                "failures": [],
            })]
        )
        # Mock _run_tests to return no runner found
        tester = TesterAgent(client=client)
        tester._run_tests = lambda: "No test runner found."  # type: ignore
        # But _run_tests is async, so we need to mock properly

        async def mock_run_tests():
            return "No test runner found."

        tester._run_tests = mock_run_tests  # type: ignore
        result = await tester.run("some goal", [])
        assert isinstance(result, TestResult)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_valid_json_response(self):
        response = json.dumps({
            "passed": True,
            "summary": "All 5 tests passed",
            "failures": [],
        })
        client = MockLLMClient(responses=[response])
        tester = TesterAgent(client=client)
        tester._run_tests = _async_no_runner  # type: ignore
        result = await tester.run("goal", [])
        assert result.passed is True
        assert "5 tests" in result.summary

    @pytest.mark.asyncio
    async def test_failed_tests(self):
        response = json.dumps({
            "passed": False,
            "summary": "3 tests failed",
            "failures": ["test_a: assertion error", "test_b: timeout"],
        })
        client = MockLLMClient(responses=[response])
        tester = TesterAgent(client=client)
        tester._run_tests = _async_no_runner  # type: ignore
        result = await tester.run("goal", [])
        assert result.passed is False
        assert len(result.failures) == 2
        assert "test_a" in result.failures[0]

    @pytest.mark.asyncio
    async def test_invalid_json_fallback(self):
        client = MockLLMClient(responses=["Tests passed! All good."])
        tester = TesterAgent(client=client)
        tester._run_tests = _async_no_runner  # type: ignore
        result = await tester.run("goal", [])
        assert result.passed is True
        assert "Tests" in result.summary

    @pytest.mark.asyncio
    async def test_test_exception_returns_failed(self):
        client = MockLLMClient(responses=[])
        tester = TesterAgent(client=client)

        async def raise_error():
            raise RuntimeError("Test runner crashed")

        tester._run_tests = raise_error  # type: ignore
        result = await tester.run("goal", [])
        assert result.passed is False
        assert "crashed" in result.summary


async def _async_no_runner() -> str:
    return "No test runner found."
