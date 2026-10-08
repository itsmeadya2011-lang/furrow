import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from furrow.agents.worker import WorkerAgent
from furrow.agents.tester import TesterAgent
from furrow.config import TaskModel, Settings
from furrow.llm import LLMClient


@pytest.fixture
def mock_client():
    client = MagicMock(spec=LLMClient)
    client.settings = Settings()
    client.complete = AsyncMock(return_value="Task completed successfully")
    client.read_file = AsyncMock(return_value="file content")
    client.write_file = AsyncMock()
    client.list_files = MagicMock(return_value=["file1.py", "file2.py"])
    return client


@pytest.mark.asyncio
async def test_worker_agent_basic(mock_client):
    task = TaskModel(id="1", description="Create a hello world function", files=["hello.py"])
    worker = WorkerAgent(task=task, client=mock_client, max_rounds=1)
    
    result = await worker.run()
    
    assert "Task completed successfully" in result
    mock_client.complete.assert_called_once()


@pytest.mark.asyncio
async def test_worker_agent_with_tool_calls(mock_client):
    # Mock response with tool calls
    mock_client.complete = AsyncMock(side_effect=[
        '```tool\n{"name": "write_file", "arguments": {"path": "hello.py", "content": "print(1)"}}\n```',
        "File created successfully",
    ])
    mock_client.read_file = AsyncMock(return_value="")
    mock_client.write_file = AsyncMock()
    
    task = TaskModel(id="1", description="Create a hello world function", files=["hello.py"])
    worker = WorkerAgent(task=task, client=mock_client, max_rounds=2)
    
    result = await worker.run()
    
    assert mock_client.write_file.called
    mock_client.complete.assert_called()


@pytest.mark.asyncio
async def test_tester_agent_tests_passed(mock_client):
    mock_client.complete = AsyncMock(return_value='{"passed": true, "summary": "All tests passed", "failures": []}')
    
    # Mock _run_tests to return passing output
    with patch.object(TesterAgent, '_run_tests', AsyncMock(return_value="5 passed")):
        tester = TesterAgent(client=mock_client)
        result = await tester.run("Test goal", [])
    
    assert result.passed is True
    assert result.summary == "All tests passed"


@pytest.mark.asyncio
async def test_tester_agent_tests_failed(mock_client):
    mock_client.complete = AsyncMock(return_value='{"passed": false, "summary": "Tests failed", "failures": ["test1 failed"]}')
    
    with patch.object(TesterAgent, '_run_tests', AsyncMock(return_value="1 failed, 4 passed")):
        tester = TesterAgent(client=mock_client, max_fix_rounds=0)
        result = await tester.run("Test goal", [])
    
    assert result.passed is False
    assert len(result.failures) == 1