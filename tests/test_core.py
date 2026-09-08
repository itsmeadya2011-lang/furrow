import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.prompts import PLANNER_PROMPT, TESTER_PROMPT, WORKER_PROMPT
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    task = TaskModel(id="1", description="do thing")
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_task_model_with_values():
    task = TaskModel(id="2", description="do thing", files=["a.py"], dependencies=["1"], status="completed", result="done")
    assert task.files == ["a.py"]
    assert task.dependencies == ["1"]
    assert task.status == "completed"
    assert task.result == "done"


def test_planner_prompt():
    assert "planning agent" in PLANNER_PROMPT.lower()
    assert "JSON" in PLANNER_PROMPT


def test_worker_prompt():
    assert "worker agent" in WORKER_PROMPT.lower()
    assert "concise" in WORKER_PROMPT.lower()


def test_tester_prompt():
    assert "tester agent" in TESTER_PROMPT.lower()
    assert "JSON" in TESTER_PROMPT


def test_settings_defaults():
    settings = Settings()
    assert settings.provider == Provider.ANTHROPIC
    assert settings.max_parallel_tasks == 5
    assert settings.max_cycles == 0
    assert settings.workspace == Path.cwd()


def test_planner_agent_parses_json():
    mock_client = MagicMock(spec=LLMClient)
    mock_client.complete = AsyncMock(return_value=json.dumps({
        "tasks": [
            {"id": "1", "description": "task one", "files": ["a.py"], "dependencies": []}
        ],
        "rationale": "plan"
    }))
    mock_client.settings = Settings()
    
    agent = PlannerAgent(client=mock_client)
    plan = asyncio.run(agent.plan("do something"))
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "task one"
    assert plan.tasks[0].files == ["a.py"]


def test_planner_agent_raises_on_invalid_json():
    mock_client = MagicMock(spec=LLMClient)
    mock_client.complete = AsyncMock(return_value="not json")
    mock_client.settings = Settings()
    
    agent = PlannerAgent(client=mock_client)
    with pytest.raises(ValueError, match="Failed to parse plan"):
        asyncio.run(agent.plan("do something"))


def test_worker_agent_returns_result():
    mock_client = MagicMock(spec=LLMClient)
    mock_client.complete = AsyncMock(return_value="implemented auth")
    mock_client.settings = Settings()
    
    task = TaskModel(id="1", description="implement auth", files=["auth.py"])
    agent = WorkerAgent(task=task, client=mock_client)
    result = asyncio.run(agent.run())
    assert result == "implemented auth"


def test_worker_agent_prompt_includes_task_details():
    mock_client = MagicMock(spec=LLMClient)
    mock_client.complete = AsyncMock(return_value="done")
    mock_client.settings = Settings()
    
    task = TaskModel(id="1", description="implement auth", files=["auth.py", "test_auth.py"])
    agent = WorkerAgent(task=task, client=mock_client)
    asyncio.run(agent.run())
    call_args = mock_client.complete.call_args
    assert "implement auth" in call_args[0][0]
    assert "auth.py" in call_args[0][0]
    assert "test_auth.py" in call_args[0][0]


def test_tester_agent_passes_when_output_contains_passed():
    mock_client = MagicMock(spec=LLMClient)
    mock_client.complete = AsyncMock(return_value=json.dumps({
        "passed": True,
        "summary": "All tests passed",
        "failures": []
    }))
    mock_client.settings = Settings()
    
    agent = TesterAgent(client=mock_client)
    result = asyncio.run(agent.run("goal", []))
    assert result.passed is True


def test_tester_agent_fallback_when_response_contains_passed():
    mock_client = MagicMock(spec=LLMClient)
    mock_client.complete = AsyncMock(return_value="All tests passed")
    mock_client.settings = Settings()
    
    agent = TesterAgent(client=mock_client)
    result = asyncio.run(agent._parse_result("All tests passed"))
    assert result.passed is True


@pytest.mark.asyncio
async def test_tester_run_tests_no_runner():
    with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
        agent = TesterAgent()
        result = await agent._run_tests()
        assert "No test runner found" in result


def test_orchestrator_initialization():
    orchestrator = Orchestrator(goal="test goal")
    assert orchestrator.goal == "test goal"
    assert orchestrator.cycles == 0
    assert orchestrator.plan is None


def test_orchestrator_is_done_with_no_plan():
    orchestrator = Orchestrator(goal="test")
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_with_completed_tasks():
    orchestrator = Orchestrator(goal="test")
    orchestrator.plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="completed"),
        ],
        rationale="ok"
    )
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_with_pending_tasks():
    orchestrator = Orchestrator(goal="test")
    orchestrator.plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="pending"),
        ],
        rationale="ok"
    )
    assert orchestrator._is_done() is False


def test_orchestrator_is_done_with_failed_tasks():
    orchestrator = Orchestrator(goal="test")
    orchestrator.plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="failed"),
        ],
        rationale="ok"
    )
    assert orchestrator._is_done() is False


def test_orchestrator_max_cycles():
    settings = Settings(max_cycles=2)
    orchestrator = Orchestrator(goal="test", client=LLMClient(settings=settings))
    assert orchestrator.client.settings.max_cycles == 2


def test_llm_client_read_write_file(tmp_path: Path):
    settings = Settings()
    client = LLMClient(settings=settings)
    
    test_file = tmp_path / "test.txt"
    asyncio.run(client.write_file(test_file, "hello world"))
    assert test_file.read_text() == "hello world"
    
    content = asyncio.run(client.read_file(test_file))
    assert content == "hello world"


def test_llm_client_list_files(tmp_path: Path):
    settings = Settings()
    client = LLMClient(settings=settings)
    
    (tmp_path / "a.py").write_text("print(1)")
    (tmp_path / "b.py").write_text("print(2)")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.py").write_text("print(3)")
    
    files = client.list_files(tmp_path)
    assert "a.py" in files
    assert "b.py" in files
    assert "sub/c.py" in files


def test_llm_client_list_files_nonexistent():
    settings = Settings()
    client = LLMClient(settings=settings)
    files = client.list_files("/nonexistent/path/12345")
    assert files == []
