from __future__ import annotations

import json
from unittest.mock import AsyncMock

from furrow.agents.tester import TesterAgent
from furrow.config import TestResult
from tests.conftest import MockLLMClient


def _tester(responses: list[str]) -> TesterAgent:
    tester = TesterAgent(client=MockLLMClient(responses=responses))
    tester._run_tests = AsyncMock(return_value="")  # avoid subprocess
    return tester


class TestTester:
    async def test_run_passed(self):
        resp = json.dumps({"passed": True, "summary": "all good", "failures": []})
        tester = _tester([resp])
        result = await tester.run("goal", [])
        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.summary == "all good"
        assert result.failures == []

    async def test_run_failed(self):
        resp = json.dumps(
            {"passed": False, "summary": "2 tests failed", "failures": ["test_a", "test_b"]}
        )
        tester = _tester([resp])
        result = await tester.run("goal", [])
        assert result.passed is False
        assert len(result.failures) == 2
        assert "test_a" in result.failures

    async def test_json_parsing_fallback_pass(self):
        tester = _tester(["tests passed and everything is passed"])
        result = await tester.run("goal", [])
        assert result.passed is True
        assert "passed" in result.summary.lower()
        assert result.failures == []

    async def test_json_parsing_fallback_fail(self):
        tester = _tester(["tests failed here"])
        result = await tester.run("goal", [])
        assert result.passed is False
        assert "tests failed here" == result.summary

    async def test_run_tests_invoked(self):
        resp = json.dumps({"passed": True, "summary": "ok", "failures": []})
        tester = _tester([resp])
        await tester.run("goal", [])
        tester._run_tests.assert_awaited_once()

    async def test_run_tests_exception_returns_failure(self):
        tester = TesterAgent(client=MockLLMClient(responses=[]))
        tester._run_tests = AsyncMock(side_effect=RuntimeError("no runner"))
        result = await tester.run("goal", [])
        assert result.passed is False
        assert "no runner" in result.summary
        assert result.failures  # contains the error

    async def test_passes_goal_and_output_in_prompt(self):
        tester = _tester([json.dumps({"passed": True, "summary": "ok", "failures": []})])
        tester._run_tests = AsyncMock(return_value="SOME_TEST_OUTPUT")
        client = tester.client
        await tester.run("a goal", [])
        prompt = client.calls[-1]["prompt"]
        assert "a goal" in prompt
        assert "SOME_TEST_OUTPUT" in prompt
