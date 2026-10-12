import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from furrow.core.orchestrator import Orchestrator, run_task_with_retry
from furrow.config import TaskModel, Plan, TestResult
from furrow.llm import LLMClient


@pytest.fixture
def mock_client():
    import os
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    client = LLMClient()
    yield client
    del os.environ["ANTHROPIC_API_KEY"]


@pytest.mark.asyncio
async def test_orchestrator_initialization(mock_client):
    orchestrator = Orchestrator(goal="Test goal", client=mock_client)
    assert orchestrator.goal == "Test goal"
    assert orchestrator.cycles == 0
    assert orchestrator.all_tasks == []


@pytest.mark.asyncio
async def test_orchestrator_is_done_empty():
    import os
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    try:
        client = LLMClient()
        orchestrator = Orchestrator(goal="Test goal", client=client)
        assert orchestrator._is_done() is False
    finally:
        del os.environ["ANTHROPIC_API_KEY"]


@pytest.mark.asyncio
async def test_orchestrator_is_done_with_completed_tasks():
    import os
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    try:
        client = LLMClient()
        orchestrator = Orchestrator(goal="Test goal", client=client)
        
        task1 = TaskModel(id="1", description="Task 1", status="completed")
        task2 = TaskModel(id="2", description="Task 2", status="completed")
        orchestrator.all_tasks = [task1, task2]
        
        assert orchestrator._is_done() is True
    finally:
        del os.environ["ANTHROPIC_API_KEY"]


@pytest.mark.asyncio
async def test_orchestrator_is_done_with_failed_tasks():
    import os
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    try:
        client = LLMClient()
        orchestrator = Orchestrator(goal="Test goal", client=client)
        
        task1 = TaskModel(id="1", description="Task 1", status="completed")
        task2 = TaskModel(id="2", description="Task 2", status="failed")
        orchestrator.all_tasks = [task1, task2]
        
        assert orchestrator._is_done() is False
    finally:
        del os.environ["ANTHROPIC_API_KEY"]


@pytest.mark.asyncio
async def test_orchestrator_is_done_with_pending_tasks():
    import os
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    try:
        client = LLMClient()
        orchestrator = Orchestrator(goal="Test goal", client=client)
        
        task1 = TaskModel(id="1", description="Task 1", status="completed")
        task2 = TaskModel(id="2", description="Task 2", status="pending")
        orchestrator.all_tasks = [task1, task2]
        
        assert orchestrator._is_done() is False
    finally:
        del os.environ["ANTHROPIC_API_KEY"]


@pytest.mark.asyncio
async def test_run_task_with_retry_success():
    import os
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    try:
        client = LLMClient()
        task = TaskModel(id="1", description="Test task")
        
        # Mock the WorkerAgent.run to return success
        with patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_class:
            mock_worker = AsyncMock()
            mock_worker.run.return_value = "Success"
            mock_worker_class.return_value = mock_worker
            
            result = await run_task_with_retry(task, client, max_retries=3)
            assert result == "Success"
            assert mock_worker.run.call_count == 1
    finally:
        del os.environ["ANTHROPIC_API_KEY"]


@pytest.mark.asyncio
async def test_run_task_with_retry_failure():
    import os
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    try:
        client = LLMClient()
        task = TaskModel(id="1", description="Test task")
        
        # Mock the WorkerAgent.run to always fail
        with patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_class:
            mock_worker = AsyncMock()
            mock_worker.run.side_effect = Exception("Test error")
            mock_worker_class.return_value = mock_worker
            
            with pytest.raises(Exception, match="Test error"):
                await run_task_with_retry(task, client, max_retries=3)
            
            # Should have retried 3 times
            assert mock_worker.run.call_count == 3
    finally:
        del os.environ["ANTHROPIC_API_KEY"]