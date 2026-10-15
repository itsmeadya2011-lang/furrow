"""Test TesterAgent on calculator.py and test_calculator.py when tests pass."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from furrow.agents.tester import TesterAgent
from furrow.config import Settings, TaskModel, TestResult


def test_tester_agent_with_passing_calculator_tests():
    """Verify TesterAgent returns expected summary when calculator tests pass."""

    async def run_test():
        # Create a mock LLM client that returns a passing TestResult
        mock_client = MagicMock()
        mock_client.settings = Settings(
            provider="anthropic",
            anthropic_api_key="fake-key",
            openai_api_key="fake-key",
        )
        mock_client.settings.tester_model = "claude-3-5-sonnet-20241022"
        mock_client.complete = AsyncMock(
            return_value=json.dumps(
                {
                    "passed": True,
                    "summary": "All 5 tests passed successfully. No failures detected.",
                    "failures": [],
                }
            )
        )

        agent = TesterAgent(client=mock_client)

        goal = "Verify the calculator implementation passes all tests"
        tasks = [
            TaskModel(
                id="1",
                description="Calculator implementation",
                files=["calculator.py"],
            ),
            TaskModel(
                id="2",
                description="Calculator tests",
                files=["tests/test_calculator.py"],
                dependencies=["calculator.py"],
            ),
        ]

        result = await agent.run(goal, tasks)

        # Assertions
        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.failures == []
        assert "passed" in result.summary.lower()
        print(f"Passed: {result.passed}")
        print(f"Summary: {result.summary}")
        print(f"Failures: {result.failures}")

        # Verify the mock was called with test output containing pytest results
        mock_client.complete.assert_called_once()
        call_args = mock_client.complete.call_args
        prompt = call_args[0][0]
        assert "Goal:" in prompt
        assert "Test output:" in prompt

        return result

    return asyncio.run(run_test())


if __name__ == "__main__":
    test_tester_agent_with_passing_calculator_tests()
    print("Test passed!")