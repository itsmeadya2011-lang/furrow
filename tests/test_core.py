import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.llm import LLMClient


# Test Config Models
def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    task = TaskModel(id="1", description="test task")
    assert task.status == "pending"
    assert task.result is None
    assert task.files == []
    assert task.dependencies == []


def test_test_result_with_failures():
    t = TestResult(
        passed=False,
        summary="2 tests failed",
        failures=["test_foo failed: assert 1 == 2", "test_bar failed: timeout"],
    )
    assert t.passed is False
    assert len(t.failures) == 2


def test_settings_defaults():
    settings = Settings()
    assert settings.provider == Provider.ANTHROPIC
    assert settings.max_parallel_tasks == 5
    assert settings.max_cycles == 0
    assert settings.workspace == Path.cwd()


# Test LLM Client
def test_llm_client_init():
    settings = Settings(provider=Provider.ANTHROPIC)
    client = LLMClient(settings=settings)
    assert client.settings == settings


def test_ollama_property():
    settings = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=settings)
    # Just check that the property doesn't raise
    client._ollama is None or True  # Will be created on first access


# Test Worker Agent
@pytest.mark.asyncio
async def test_worker_agent_run():
    settings = Settings()
    mock_client = MagicMock(spec=LLMClient)
    mock_client.settings = settings
    mock_client.complete = AsyncMock(return_value="Test response")
    mock_client.write_file = AsyncMock()
    
    task = TaskModel(id="1", description="test task", files=["test.py"])
    worker = WorkerAgent(task=task, client=mock_client)
    
    result = await worker.run()
    assert result == "Test response"
    mock_client.complete.assert_called_once()


@pytest.mark.asyncio
async def test_worker_agent_creates_file_from_json():
    settings = Settings()
    mock_client = MagicMock(spec=LLMClient)
    mock_client.settings = settings
    mock_client.complete = AsyncMock(
        return_value="""I created the file.

```json
{
  "files": [
    {
      "path": "test.py",
      "action": "create",
      "content": "print('hello')"
    }
  ]
}
```"""
    )
    mock_client.write_file = AsyncMock()
    
    task = TaskModel(id="1", description="create test file")
    worker = WorkerAgent(task=task, client=mock_client)
    
    result = await worker.run()
    mock_client.write_file.assert_called_once()


# Test Planner Agent
@pytest.mark.asyncio
async def test_planner_agent_plan():
    settings = Settings()
    mock_client = MagicMock(spec=LLMClient)
    mock_client.settings = settings
    mock_client.complete = AsyncMock(
        return_value=json.dumps({
            "tasks": [{"id": "1", "description": "task 1", "files": []}],
            "rationale": "test plan",
        })
    )
    
    planner = PlannerAgent(client=mock_client)
    plan = await planner.plan("test goal")
    
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"
    assert plan.rationale == "test plan"


# Test Tester Agent
@pytest.mark.asyncio
async def test_tester_agent_run_with_passing_tests():
    settings = Settings()
    mock_client = MagicMock(spec=LLMClient)
    mock_client.settings = settings
    mock_client.complete = AsyncMock(
        return_value=json.dumps({
            "passed": True,
            "summary": "All tests passed",
            "failures": [],
        })
    )
    
    tester = TesterAgent(client=mock_client)
    result = await tester.run("test goal", [])
    
    assert result.passed is True
    assert result.summary == "All tests passed"


@pytest.mark.asyncio
async def test_tester_agent_handles_invalid_json():
    settings = Settings()
    mock_client = MagicMock(spec=LLMClient)
    mock_client.settings = settings
    mock_client.complete = AsyncMock(return_value="Tests passed!")
    
    tester = TesterAgent(client=mock_client)
    result = await tester.run("test goal", [])
    
    assert result.passed is True
    assert result.summary == "Tests passed!"


# Test Orchestrator
@pytest.mark.asyncio
async def test_orchestrator_max_cycles():
    from furrow.core.orchestrator import Orchestrator
    
    settings = Settings(max_cycles=1)
    mock_client = MagicMock(spec=LLMClient)
    mock_client.settings = settings
    
    # Mock planner to return no tasks
    mock_planner = MagicMock()
    mock_planner.plan = AsyncMock(return_value=Plan(tasks=[], rationale="done"))
    
    orchestrator = Orchestrator(goal="test", client=mock_client)
    orchestrator.planner = mock_planner
    
    with patch("furrow.core.orchestrator.console") as mock_console:
        await orchestrator.run()
    
    # Should only run one cycle due to max_cycles=1
    assert orchestrator.cycles == 1


@pytest.mark.asyncio
async def test_orchestrator_is_done_with_empty_tasks():
    from furrow.core.orchestrator import Orchestrator
    
    orchestrator = Orchestrator(goal="test")
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_is_done_with_completed_tasks():
    from furrow.core.orchestrator import Orchestrator
    
    orchestrator = Orchestrator(goal="test")
    orchestrator.tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="completed"),
    ]
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_is_done_with_failed_tasks():
    from furrow.core.orchestrator import Orchestrator
    
    orchestrator = Orchestrator(goal="test")
    orchestrator.tasks = [
        TaskModel(id="1", description="task 1", status="failed"),
    ]
    # Should not be done if there are failed tasks (will retry)
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_get_tasks():
    from furrow.core.orchestrator import Orchestrator
    
    orchestrator = Orchestrator(goal="test")
    orchestrator.tasks = [TaskModel(id="1", description="task 1")]
    
    assert len(orchestrator._get_tasks()) == 1
    assert orchestrator._get_tasks()[0].id == "1"
