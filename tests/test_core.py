import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import OrchestratorState, Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    task = TaskModel(id="1", description="test")
    assert task.status == "pending"
    assert task.files == []
    assert task.dependencies == []
    assert task.result is None


def test_plan_with_dependencies():
    tasks = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2", dependencies=["1"]),
    ]
    p = Plan(tasks=tasks, rationale="test")
    assert p.tasks[1].dependencies == ["1"]


def test_orchestrator_state_serialization():
    tasks = [TaskModel(id="1", description="test", status="completed")]
    state = OrchestratorState(goal="test goal", cycles=3, tasks=tasks)
    json_str = state.model_dump_json()
    parsed = json.loads(json_str)
    assert parsed["goal"] == "test goal"
    assert parsed["cycles"] == 3
    assert len(parsed["tasks"]) == 1
    assert parsed["tasks"][0]["status"] == "completed"


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"


def test_settings_provider_enum():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"


@pytest.mark.asyncio
async def test_orchestrator_initialization():
    with patch("furrow.core.orchestrator.LLMClient"):
        orchestrator = Orchestrator(goal="test goal")
        assert orchestrator.goal == "test goal"
        assert orchestrator.cycles == 0
        assert orchestrator.all_tasks == []


@pytest.mark.asyncio
async def test_orchestrator_state_persistence(tmp_path: Path):
    state_file = tmp_path / "test_state.json"
    tasks = [TaskModel(id="1", description="test", status="completed")]
    
    with patch("furrow.core.orchestrator.LLMClient"):
        orchestrator = Orchestrator(goal="test goal", state_file=state_file)
        orchestrator.all_tasks = tasks
        orchestrator.cycles = 2
        orchestrator._save_state()
    
    # Load state in new orchestrator
    with patch("furrow.core.orchestrator.LLMClient"):
        orchestrator2 = Orchestrator(goal="different goal", state_file=state_file)
        assert orchestrator2.goal == "test goal"
        assert orchestrator2.cycles == 2
        assert len(orchestrator2.all_tasks) == 1
        assert orchestrator2.all_tasks[0].status == "completed"


@pytest.mark.asyncio
async def test_llm_client_initialization():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        client = LLMClient()
        assert client.settings.provider == Provider.ANTHROPIC


@pytest.mark.asyncio
async def test_llm_client_ollama_provider():
    with patch.dict("os.environ", {}):
        settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
        client = LLMClient(settings=settings)
        assert client.settings.provider == Provider.OLLAMA


@pytest.mark.asyncio
async def test_planner_agent_plan_parsing():
    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(return_value=json.dumps({
        "tasks": [{"id": "1", "description": "test task", "files": [], "dependencies": []}],
        "rationale": "test"
    }))
    mock_client.settings.planner_model = "test-model"
    
    planner = PlannerAgent(client=mock_client)
    plan = await planner.plan("test goal")
    
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "test task"
    mock_client.complete.assert_called_once()


@pytest.mark.asyncio
async def test_worker_agent_tools():
    mock_client = AsyncMock()
    mock_client.complete_with_tools = AsyncMock(return_value="Task completed successfully")
    mock_client.settings.worker_model = "test-model"
    
    task = TaskModel(id="1", description="test task", files=["test.py"])
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()
    
    assert result == "Task completed successfully"
    mock_client.complete_with_tools.assert_called_once()


@pytest.mark.asyncio
async def test_tester_agent_test_evaluation():
    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(return_value=json.dumps({
        "passed": True,
        "summary": "All tests passed",
        "failures": []
    }))
    mock_client.settings.tester_model = "test-model"
    
    tester = TesterAgent(client=mock_client)
    tasks = [TaskModel(id="1", description="test", status="completed")]
    result = await tester.run("test goal", tasks)
    
    assert result.passed is True
    assert result.summary == "All tests passed"
    mock_client.complete.assert_called_once()


@pytest.mark.asyncio
async def test_tester_agent_failed_tests():
    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(return_value=json.dumps({
        "passed": False,
        "summary": "Tests failed",
        "failures": ["test_file.py::test_func - AssertionError"]
    }))
    mock_client.settings.tester_model = "test-model"
    
    tester = TesterAgent(client=mock_client)
    tasks = [TaskModel(id="1", description="test", status="completed")]
    result = await tester.run("test goal", tasks)
    
    assert result.passed is False
    assert len(result.failures) == 1


@pytest.mark.asyncio
async def test_orchestrator_max_cycles():
    with patch("furrow.core.orchestrator.LLMClient") as mock_llm:
        mock_client = AsyncMock()
        mock_llm.return_value = mock_client
        
        settings = Settings(max_cycles=2)
        with patch("furrow.core.orchestrator.settings", settings):
            orchestrator = Orchestrator(goal="test goal")
            orchestrator.cycles = 2
            
            # Should stop at max_cycles
            assert settings.max_cycles == 2


@pytest.mark.asyncio
async def test_orchestrator_is_done():
    with patch("furrow.core.orchestrator.LLMClient"):
        orchestrator = Orchestrator(goal="test goal")
        
        # No tasks - not done
        assert orchestrator._is_done() is False
        
        # Add completed tasks
        orchestrator.all_tasks = [
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="completed"),
        ]
        assert orchestrator._is_done() is True
        
        # Add failed task
        orchestrator.all_tasks.append(TaskModel(id="3", description="task 3", status="failed"))
        assert orchestrator._is_done() is False
        
        # Add pending task
        orchestrator.all_tasks = [
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="pending"),
        ]
        assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_worker_tool_definitions():
    from furrow.agents.worker import WORKER_TOOLS
    
    assert len(WORKER_TOOLS) == 3
    tool_names = {t["name"] for t in WORKER_TOOLS}
    assert tool_names == {"read_file", "write_file", "list_files"}
    
    for tool in WORKER_TOOLS:
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"
