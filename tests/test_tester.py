"""Tests for the TesterAgent JSON parsing robustness."""

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from unittest.mock import AsyncMock, MagicMock


def make_mock_client(response_text: str, test_output: str = "all passed") -> MagicMock:
    client = MagicMock()
    client.settings = MagicMock()
    client.tester_model = "test-model"
    client.complete = AsyncMock(return_value=response_text)

    # Mock the test runner to return controlled output
    tester = TesterAgent(client=client)
    return tester, client


class TestTesterJSONParsing:
    """Tests for robust JSON parsing in TesterAgent."""

    @pytest.mark.asyncio
    async def test_valid_json_response(self) -> None:
        tester, client = make_mock_client(
            '{"passed": true, "summary": "All tests passed", "failures": []}'
        )
        # Patch _run_tests to skip actual test execution
        tester._run_tests = AsyncMock(return_value="all passed")
        result = await tester.run("test goal", [TaskModel(id="1", description="t")])
        assert result.passed is True
        assert result.summary == "All tests passed"

    @pytest.mark.asyncio
    async def test_json_wrapped_in_markdown(self) -> None:
        tester, _ = make_mock_client(
            "```json\n{\"passed\": false, \"summary\": \"3 tests failed\", \"failures\": [\"test_auth\"]}\n```"
        )
        tester._run_tests = AsyncMock(return_value="failures...")
        result = await tester.run("test goal", [])
        assert result.passed is False
        assert result.summary == "3 tests failed"
        assert result.failures == ["test_auth"]

    @pytest.mark.asyncio
    async def test_invalid_json_fallback(self) -> None:
        tester, _ = make_mock_client(
            "Sorry, I cannot parse this. The tests seem to have passed though."
        )
        tester._run_tests = AsyncMock(return_value="all passed")
        result = await tester.run("test goal", [])
        # "passed" is in the response, so fallback should set passed=True
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_none_test_runner_returns_message(self) -> None:
        tester, _ = make_mock_client("not json")
        tester._run_tests = AsyncMock(return_value="No test runner found.")
        result = await tester.run("test", [])
        assert result.passed is False  # "not json" doesn't contain "passed"
        assert "No test runner" in result.summary or "passed" in result.summary.lower() or "not json" in result.summary
