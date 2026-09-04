from unittest.mock import AsyncMock, MagicMock, patch

from furrow.agents.tester import TesterAgent
from furrow.config import TaskModel


async def test_no_test_runner():
    mock_client = MagicMock()
    agent = TesterAgent(client=mock_client)
    with patch(
        'furrow.agents.tester.asyncio.create_subprocess_exec',
        side_effect=FileNotFoundError,
    ):
        result = await agent._run_tests()
        assert result == "No test runner found."


async def test_fail_parse_returns_fallback():
    mock_settings = MagicMock()
    mock_settings.tester_model = "test-model"
    mock_client = MagicMock()
    mock_client.settings = mock_settings

    agent = TesterAgent(client=mock_client)
    tasks = [TaskModel(id="1", description="task")]

    with patch.object(agent, '_run_tests', new_callable=AsyncMock, return_value="some test output"):
        mock_client.complete = AsyncMock(return_value="tests passed successfully")
        result = await agent.run("goal", tasks)
        assert result.passed is True
        assert result.summary == "tests passed successfully"

        mock_client.complete = AsyncMock(return_value="tests failed badly")
        result = await agent.run("goal", tasks)
        assert result.passed is False
