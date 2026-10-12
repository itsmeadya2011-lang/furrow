import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from furrow.llm import LLMClient


@pytest.fixture
def mock_client():
    import os
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    client = LLMClient()
    yield client
    del os.environ["ANTHROPIC_API_KEY"]


@pytest.mark.asyncio
async def test_worker_tool_read_file(mock_client, tmp_path):
    test_file = tmp_path / "test.txt"
    test_content = "Hello, World!"
    test_file.write_text(test_content)
    
    # Change to tmp_path for workspace
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        task = TaskModel(id="1", description="Read test file", files=["test.txt"])
        worker = WorkerAgent(task=task, client=mock_client)
        
        result = await worker._tool_read_file({"path": "test.txt"})
        assert "Hello, World!" in result
    finally:
        os.chdir(old_cwd)


@pytest.mark.asyncio
async def test_worker_tool_write_file(mock_client, tmp_path):
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        task = TaskModel(id="1", description="Write test file")
        worker = WorkerAgent(task=task, client=mock_client)
        
        result = await worker._tool_write_file({"path": "new_file.txt", "content": "New content"})
        assert "Successfully wrote" in result
        
        # Verify file was created
        new_file = tmp_path / "new_file.txt"
        assert new_file.exists()
        assert new_file.read_text() == "New content"
    finally:
        os.chdir(old_cwd)


@pytest.mark.asyncio
async def test_worker_tool_list_files(mock_client, tmp_path):
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    # Create some test files
    (tmp_path / "file1.txt").write_text("1")
    (tmp_path / "file2.txt").write_text("2")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file3.txt").write_text("3")
    
    try:
        task = TaskModel(id="1", description="List files")
        worker = WorkerAgent(task=task, client=mock_client)
        
        result = await worker._tool_list_files({"directory": "."})
        assert "file1.txt" in result
        assert "file2.txt" in result
    finally:
        os.chdir(old_cwd)


@pytest.mark.asyncio
async def test_worker_tool_run_command(mock_client, tmp_path):
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        task = TaskModel(id="1", description="Run command")
        worker = WorkerAgent(task=task, client=mock_client)
        
        # Test echo command
        result = await worker._tool_run_command({"command": "echo hello"})
        assert "hello" in result
        assert "Exit code: 0" in result
    finally:
        os.chdir(old_cwd)


@pytest.mark.asyncio
async def test_worker_tool_run_command_not_allowed(mock_client):
    task = TaskModel(id="1", description="Run command")
    worker = WorkerAgent(task=task, client=mock_client)
    
    result = await worker._tool_run_command({"command": "rm -rf /"})
    assert "Command not allowed" in result


@pytest.mark.asyncio
async def test_worker_parse_tool_calls():
    import os
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    try:
        client = LLMClient()
        task = TaskModel(id="1", description="Test")
        worker = WorkerAgent(task=task, client=client)
        
        response = '''
Some text before
```tool read_file {"path": "test.txt"}
```
Some text after
'''
        tool_calls = worker._parse_tool_calls(response)
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "read_file"
        assert tool_calls[0]["arguments"]["path"] == "test.txt"
    finally:
        del os.environ["ANTHROPIC_API_KEY"]