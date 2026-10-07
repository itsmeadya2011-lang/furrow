import pytest
from unittest.mock import AsyncMock, patch

from furrow.agents.tester import TesterAgent
from furrow.config import TaskModel, TestResult


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_run_returns_passed_true_when_tests_pass(self):
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            return_value='{"passed": true, "summary": "All tests passed", "failures": []}'
        )
        mock_client.settings.tester_model = "test-model"

        agent = TesterAgent(client=mock_client)

        with patch.object(agent, "_run_tests", new_callable=AsyncMock, return_value="all tests passed"):
            result = await agent.run("build feature", [])

        assert isinstance(result, TestResult)
        assert result.passed is True
        assert "passed" in result.summary.lower()

    @pytest.mark.asyncio
    async def test_run_returns_passed_false_when_tests_fail(self):
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            return_value='{"passed": false, "summary": "Tests failed", "failures": ["test_a.py::test_b AssertionError"]}'
        )
        mock_client.settings.tester_model = "test-model"

        agent = TesterAgent(client=mock_client)

        with patch.object(agent, "_run_tests", new_callable=AsyncMock, return_value="1 failed"):
            result = await agent.run("build feature", [])

        assert isinstance(result, TestResult)
        assert result.passed is False
        assert len(result.failures) == 1

    @pytest.mark.asyncio
    async def test_run_fallback_passed_true_on_keyword(self):
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            return_value="The tests passed successfully"
        )
        mock_client.settings.tester_model = "test-model"

        agent = TesterAgent(client=mock_client)

        with patch.object(agent, "_run_tests", new_callable=AsyncMock, return_value="pass"):
            result = await agent.run("build feature", [])

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_run_fallback_passed_false_on_keyword(self):
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            return_value="The tests failed with errors"
        )
        mock_client.settings.tester_model = "test-model"

        agent = TesterAgent(client=mock_client)

        with patch.object(agent, "_run_tests", new_callable=AsyncMock, return_value="fail"):
            result = await agent.run("build feature", [])

        assert result.passed is False

    @pytest.mark.asyncio
    async def test_run_exception_during_test_run_returns_failed(self):
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            return_value='{"passed": true, "summary": "ok", "failures": []}'
        )
        mock_client.settings.tester_model = "test-model"

        agent = TesterAgent(client=mock_client)

        with patch.object(agent, "_run_tests", new_callable=AsyncMock, side_effect=RuntimeError("pytest not found")):
            result = await agent.run("build feature", [])

        assert isinstance(result, TestResult)
        assert result.passed is False
