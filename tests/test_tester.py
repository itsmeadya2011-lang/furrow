import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import Settings, TaskModel, TestResult


class MockLLMClient:
    """A mock LLM client for testing."""

    def __init__(self, responses=None):
        self.settings = Settings()
        self.responses = responses or []

    async def complete(self, prompt, system="", model=None):
        if self.responses:
            return self.responses.pop(0)
        return "{}"


async def test_tester_run_tests_passed():
    client = MockLLMClient()
    client.responses = [
        json.dumps(
            {
                "passed": True,
                "summary": "All 5 tests passed",
                "failures": [],
            }
        )
    ]

    tester = TesterAgent(client=client)
    # Mock _run_tests to return passing output
    tester._run_tests = AsyncMock(return_value="5 passed, 0 failed")

    result = await tester.run(goal="test goal", tasks=[])
    assert result.passed is True
    assert "5 tests passed" in result.summary
    assert result.failures == []


async def test_tester_run_tests_failed():
    client = MockLLMClient()
    client.responses = [
        json.dumps(
            {
                "passed": False,
                "summary": "2 tests failed",
                "failures": [
                    "test_foo: assertion error",
                    "test_bar: timeout",
                ],
            }
        )
    ]

    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(return_value="2 failed, 3 passed")

    result = await tester.run(goal="test goal", tasks=[])
    assert result.passed is False
    assert "2 tests failed" in result.summary
    assert len(result.failures) == 2


async def test_tester_no_test_runner():
    client = MockLLMClient()
    client.responses = [
        json.dumps(
            {
                "passed": False,
                "summary": "No test runner found",
                "failures": ["No test runner available"],
            }
        )
    ]

    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(return_value="No test runner found.")

    result = await tester.run(goal="test goal", tasks=[])
    assert result.passed is False
    assert "No test runner" in result.summary


async def test_tester_invalid_json_fallback():
    client = MockLLMClient()
    client.responses = ["The tests appear to have passed based on the output."]

    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(return_value="All good")

    result = await tester.run(goal="test goal", tasks=[])
    # Fallback should detect "passed" in the response
    assert result.passed is True


async def test_tester_run_tests_exception():
    client = MockLLMClient()
    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(side_effect=RuntimeError("Test runner crashed"))

    result = await tester.run(goal="test goal", tasks=[])
    assert result.passed is False
    assert "Test runner crashed" in result.summary