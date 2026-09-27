from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TestResult


async def test_planner_raises_value_error_after_three_failed_parse():
    client = MagicMock()
    client.complete = AsyncMock(return_value="not valid json")
    client.settings.planner_model = "model"
    agent = PlannerAgent(client=client)
    with pytest.raises(ValueError):
        await agent.plan("build a thing")
    assert client.complete.await_count == 3


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("pyproject.toml", ["pytest", "-q"]),
        ("setup.cfg", ["python", "-m", "pytest", "-q"]),
        ("package.json", ["npm", "test", "--silent"]),
        ("Cargo.toml", ["cargo", "test"]),
        ("go.mod", ["go", "test", "./..."]),
    ],
)
def test_detect_test_command_for_project_types(filename, expected, tmp_path):
    (tmp_path / filename).write_text("...")
    client = MagicMock()
    client.settings.workspace = tmp_path
    agent = TesterAgent(client=client)
    assert agent._detect_test_command() == expected


def test_detect_test_command_returns_none_when_not_found(tmp_path):
    client = MagicMock()
    client.settings.workspace = tmp_path
    agent = TesterAgent(client=client)
    assert agent._detect_test_command() is None


async def test_worker_uses_complete_with_tools():
    client = MagicMock()
    client.complete_with_tools = AsyncMock(return_value="done")
    client.complete = AsyncMock()
    client.settings.worker_model = "model"
    client.settings.max_tool_calls = 30
    agent = WorkerAgent(task=TaskModel(id="1", description="a"), client=client)
    result = await agent.run()
    assert result == "done"
    client.complete_with_tools.assert_awaited_once()
    client.complete.assert_not_awaited()


@pytest.mark.parametrize(
    ("response", "passed"),
    [("All tests passed!", True), ("tests failed here", False)],
)
async def test_tester_fallback_parsing_invalid_json(response, passed):
    client = MagicMock()
    client.complete = AsyncMock(return_value=response)
    client.settings.tester_model = "model"
    agent = TesterAgent(client=client)
    agent._run_tests = AsyncMock(return_value="output")
    result = await agent.run("goal", [TaskModel(id="1", description="a")])
    assert isinstance(result, TestResult)
    assert result.passed is passed
    assert result.summary == response
    assert result.failures == []
