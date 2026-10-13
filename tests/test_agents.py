from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


class MockLLMClient:
    def __init__(self):
        self.settings = MagicMock()
        self.settings.planner_model = "test-model"
        self.settings.worker_model = "test-model"
        self.settings.tester_model = "test-model"
        self.complete = AsyncMock()


@pytest.mark.asyncio
async def test_planner_agent():
    mock_client = MockLLMClient()
    mock_client.complete.return_value = '{"tasks": [{"id": "1", "description": "test task", "files": [], "dependencies": []}], "rationale": "test"}'
    
    planner = PlannerAgent(client=mock_client)
    plan = await planner.plan("test goal")
    
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "test task"
    assert plan.rationale == "test"
    mock_client.complete.assert_called_once()


@pytest.mark.asyncio
async def test_planner_agent_parse_error():
    mock_client = MockLLMClient()
    mock_client.complete.return_value = "invalid json"
    
    planner = PlannerAgent(client=mock_client)
    
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("test goal")


@pytest.mark.asyncio
async def test_worker_agent():
    mock_client = MockLLMClient()
    mock_client.complete.return_value = "Task completed successfully"
    
    task = TaskModel(id="1", description="test task", files=["a.py"])
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()
    
    assert result == "Task completed successfully"
    mock_client.complete.assert_called_once()


@pytest.mark.asyncio
async def test_tester_agent_no_tests():
    mock_client = MockLLMClient()
    mock_client.complete.return_value = '{"passed": true, "summary": "no tests", "failures": []}'
    
    tester = TesterAgent(client=mock_client)
    tester._run_tests = AsyncMock(return_value="")
    
    result = await tester.run("test goal", [])
    
    assert result.passed is True
    assert result.summary == "no tests"


@pytest.mark.asyncio
async def test_tester_agent_with_mocked_tests():
    mock_client = MockLLMClient()
    mock_client.complete.return_value = '{"passed": false, "summary": "test failed", "failures": ["error"]}'
    
    tester = TesterAgent(client=mock_client)
    
    # Mock _run_tests to return a test output
    tester._run_tests = AsyncMock(return_value="FAILED test_example")
    
    tasks = [TaskModel(id="1", description="test task")]
    result = await tester.run("test goal", tasks)
    
    assert result.passed is False
    assert "test failed" in result.summary
    assert len(result.failures) == 1