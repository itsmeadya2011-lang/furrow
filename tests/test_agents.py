import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.call_count = 0
        self.last_prompt = ""
        self.written_files = {}
        
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        # Return response based on prompt content
        if "planner" in prompt.lower() or "plan" in prompt.lower():
            return self.responses.get("plan", json.dumps({
                "tasks": [
                    {"id": "1", "description": "Create test file", "files": ["test.txt"], "dependencies": []}
                ],
                "rationale": "Simple test plan"
            }))
        elif "worker" in prompt.lower() or "task" in prompt.lower():
            return self.responses.get("worker", json.dumps({
                "files": [{"path": "test.txt", "operation": "write", "content": "Hello World"}]
            }))
        elif "tester" in prompt.lower() or "test" in prompt.lower():
            return self.responses.get("tester", json.dumps({
                "passed": True,
                "summary": "All tests passed",
                "failures": []
            }))
        return self.responses.get("default", "OK")
    
    async def read_file(self, path: str) -> str:
        """Mock read_file method - reads actual file."""
        import aiofiles
        try:
            async with aiofiles.open(path, "r") as f:
                return await f.read()
        except Exception:  # noqa: BLE001
            return ""
    
    async def write_file(self, path: str, content: str) -> None:
        """Mock write_file method - writes actual file."""
        from pathlib import Path

        import aiofiles
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)
        self.written_files[path] = content


@pytest.fixture
def mock_client():
    return MockLLMClient()


@pytest.mark.asyncio
async def test_planner_agent_creates_plan(mock_client):
    """Test that planner agent creates a valid plan."""
    planner = PlannerAgent(client=mock_client)
    plan = await planner.plan("Create a simple test file")
    
    assert isinstance(plan, Plan)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"
    assert plan.tasks[0].description == "Create test file"
    assert plan.tasks[0].files == ["test.txt"]


@pytest.mark.asyncio
async def test_planner_agent_handles_invalid_json(mock_client):
    """Test that planner agent handles invalid JSON from LLM."""
    mock_client.responses["plan"] = "not valid json"
    planner = PlannerAgent(client=mock_client)
    
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("Create a test file")


@pytest.mark.asyncio
async def test_worker_agent_executes_file_operations(mock_client, tmp_path):
    """Test that worker agent executes file operations."""
    # Change to temp directory
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        task = TaskModel(id="1", description="Create test file", files=["test.txt"])
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()
        
        assert "Created/updated: test.txt" in result
        # Verify file was created
        assert (tmp_path / "test.txt").exists()
        assert (tmp_path / "test.txt").read_text() == "Hello World"
    finally:
        os.chdir(old_cwd)


@pytest.mark.asyncio
async def test_worker_agent_reads_context_files(mock_client, tmp_path):
    """Test that worker agent reads context files before executing."""
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    # Create a context file
    (tmp_path / "existing.py").write_text("print('hello')")
    
    try:
        task = TaskModel(id="1", description="Modify existing file", files=["existing.py"])
        worker = WorkerAgent(task=task, client=mock_client)
        await worker.run()
        
        # Verify the context was read (prompt should contain file content)
        assert "print('hello')" in mock_client.last_prompt
    finally:
        os.chdir(old_cwd)


@pytest.mark.asyncio
async def test_worker_agent_handles_non_json_response(mock_client, tmp_path):
    """Test that worker agent handles non-JSON responses gracefully."""
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    mock_client.responses["worker"] = "Just a text response"
    
    try:
        task = TaskModel(id="1", description="Do something", files=[])
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()
        
        assert result == "Just a text response"
    finally:
        os.chdir(old_cwd)


@pytest.mark.asyncio
async def test_tester_agent_runs_tests_and_parses_result(mock_client):
    """Test that tester agent runs tests and parses results."""
    from unittest.mock import AsyncMock
    tester = TesterAgent(client=mock_client)
    tester._run_tests = AsyncMock(return_value="tests passed")
    tasks = [TaskModel(id="1", description="Test task", files=["test.py"])]
    
    result = await tester.run("Test goal", tasks)
    
    assert isinstance(result, TestResult)
    assert result.passed is True
    assert result.summary == "All tests passed"


@pytest.mark.asyncio
async def test_tester_agent_handles_test_failures(mock_client):
    """Test that tester agent handles test failures."""
    from unittest.mock import AsyncMock
    mock_client.responses["tester"] = json.dumps({
        "passed": False,
        "summary": "Tests failed",
        "failures": ["test_example failed: AssertionError"]
    })
    
    tester = TesterAgent(client=mock_client)
    tester._run_tests = AsyncMock(return_value="tests failed")
    tasks = [TaskModel(id="1", description="Test task", files=["test.py"])]
    
    result = await tester.run("Test goal", tasks)
    
    assert result.passed is False
    assert len(result.failures) == 1
    assert "test_example failed" in result.failures[0]


@pytest.mark.asyncio
async def test_orchestrator_runs_cycle(mock_client, tmp_path):
    """Test that orchestrator runs a complete cycle."""
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        orchestrator = Orchestrator(goal="Create a test file", client=mock_client)
        orchestrator.cycles = 0  # Reset
        
        # Run one cycle
        await orchestrator._cycle()
        
        # Verify tasks were created and executed
        assert len(orchestrator.all_tasks) == 1
        assert orchestrator.all_tasks[0].status == "completed"
        assert (tmp_path / "test.txt").exists()
    finally:
        os.chdir(old_cwd)


@pytest.mark.asyncio
async def test_orchestrator_tracks_tasks_across_cycles(mock_client, tmp_path):
    """Test that orchestrator tracks tasks across multiple cycles."""
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # First cycle creates one task
        mock_client.responses["plan"] = json.dumps({
            "tasks": [
                {"id": "1", "description": "Task 1", "files": ["task1.txt"], "dependencies": []},
                {"id": "2", "description": "Task 2", "files": ["task2.txt"], "dependencies": []}
            ],
            "rationale": "Two tasks"
        })
        
        orchestrator = Orchestrator(goal="Create two files", client=mock_client)
        await orchestrator._cycle()
        
        assert len(orchestrator.all_tasks) == 2
        
        # Second cycle should not duplicate tasks
        mock_client.responses["plan"] = json.dumps({
            "tasks": [
                {"id": "1", "description": "Task 1", "files": ["task1.txt"], "dependencies": []},
                {"id": "2", "description": "Task 2", "files": ["task2.txt"], "dependencies": []}
            ],
            "rationale": "Same tasks"
        })
        
        await orchestrator._cycle()
        
        assert len(orchestrator.all_tasks) == 2  # Still 2, not 4
    finally:
        os.chdir(old_cwd)


@pytest.mark.asyncio
async def test_orchestrator_is_done_logic(mock_client, tmp_path):
    """Test orchestrator completion logic."""
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        orchestrator = Orchestrator(goal="Test", client=mock_client)
        
        # No tasks - not done
        assert orchestrator._is_done() is False
        
        # Add completed task
        orchestrator.all_tasks = [
            TaskModel(id="1", description="Task 1", status="completed")
        ]
        assert orchestrator._is_done() is True
        
        # Add failed task - not done
        orchestrator.all_tasks = [
            TaskModel(id="1", description="Task 1", status="completed"),
            TaskModel(id="2", description="Task 2", status="failed")
        ]
        assert orchestrator._is_done() is False
        
        # Add pending task - not done
        orchestrator.all_tasks = [
            TaskModel(id="1", description="Task 1", status="completed"),
            TaskModel(id="2", description="Task 2", status="pending")
        ]
        assert orchestrator._is_done() is False
    finally:
        os.chdir(old_cwd)


@pytest.mark.asyncio
async def test_orchestrator_handles_task_failures(mock_client, tmp_path):
    """Test that orchestrator handles task failures properly."""
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    # Make worker fail by raising a specific error for worker prompts
    original_complete = mock_client.complete
    async def failing_complete(prompt: str, system: str = "", model: str | None = None) -> str:
        if "You are a worker agent" in prompt:
            raise RuntimeError("Worker error")
        return await original_complete(prompt, system, model)
    mock_client.complete = failing_complete
    
    try:
        orchestrator = Orchestrator(goal="Test failure handling", client=mock_client)
        await orchestrator._cycle()
        
        assert len(orchestrator.all_tasks) == 1
        assert orchestrator.all_tasks[0].status == "failed"
        assert "Worker error" in orchestrator.all_tasks[0].result
    finally:
        os.chdir(old_cwd)


@pytest.mark.asyncio
async def test_llm_client_retry_logic():
    """Test that LLM client retries on failure."""
    import httpx

    from furrow.config import Provider, Settings
    
    settings = Settings(
        provider=Provider.ANTHROPIC,
        anthropic_api_key="test-key"
    )
    
    client = LLMClient(settings=settings)
    
    # Mock the anthropic client to fail twice then succeed
    mock_anthropic = AsyncMock()
    mock_anthropic.messages.create.side_effect = [
        httpx.NetworkError("Network error"),
        httpx.TimeoutException("Timeout"),
        MagicMock(content=[MagicMock(text="Success")])
    ]
    client._anthropic = mock_anthropic
    
    result = await client.complete("Test prompt")
    
    assert result == "Success"
    assert mock_anthropic.messages.create.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])