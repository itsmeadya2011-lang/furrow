import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


@pytest.mark.asyncio
async def test_planner_parses_valid_json():
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value=json.dumps({
        "tasks": [
            {"id": "1", "description": "build auth", "files": ["auth.py"], "dependencies": []}
        ],
        "rationale": "start with auth"
    }))
    mock_client.settings.planner_model = "test-model"

    planner = PlannerAgent(client=mock_client)
    plan = await planner.plan("Add auth")
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"
    assert plan.rationale == "start with auth"


@pytest.mark.asyncio
async def test_planner_raises_on_invalid_json():
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value="not json")
    mock_client.settings.planner_model = "test-model"

    planner = PlannerAgent(client=mock_client)
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("Add auth")


@pytest.mark.asyncio
async def test_worker_formats_prompt():
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value="done")
    mock_client.settings.worker_model = "test-model"

    task = TaskModel(id="1", description="do thing", files=["a.py", "b.py"], dependencies=[])
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker.run()
    assert result == "done"
    called_prompt = mock_client.complete.call_args[0][0]
    assert "do thing" in called_prompt
    assert "a.py, b.py" in called_prompt


@pytest.mark.asyncio
async def test_tester_no_runner_found():
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value='{"passed": false, "summary": "no tests", "failures": []}')
    mock_client.settings.tester_model = "test-model"
    mock_client.settings.workspace = "/nonexistent"

    tester = TesterAgent(client=mock_client)
    result = await tester.run("goal", [])
    assert result.passed is False
    assert "No test runner found" in result.summary


def test_orchestrator_get_tasks_returns_plan_tasks():
    orchestrator = Orchestrator(goal="test")
    assert orchestrator._get_tasks() == []

    mock_plan = Plan(tasks=[TaskModel(id="1", description="task")], rationale="ok")
    orchestrator.plan = mock_plan
    assert len(orchestrator._get_tasks()) == 1


def test_orchestrator_is_done_logic():
    orchestrator = Orchestrator(goal="test")
    assert orchestrator._is_done() is False

    mock_plan = Plan(tasks=[
        TaskModel(id="1", description="task1", status="completed"),
        TaskModel(id="2", description="task2", status="completed"),
    ], rationale="ok")
    orchestrator.plan = mock_plan
    assert orchestrator._is_done() is True


def test_orchestrator_not_done_with_failed_tasks():
    orchestrator = Orchestrator(goal="test")
    mock_plan = Plan(tasks=[
        TaskModel(id="1", description="task1", status="completed"),
        TaskModel(id="2", description="task2", status="failed"),
    ], rationale="ok")
    orchestrator.plan = mock_plan
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_llm_client_unsupported_provider():
    from furrow.config import Settings, Provider
    settings = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=settings)

    with pytest.raises(ValueError, match="Unsupported provider"):
        # This will actually raise from _complete_anthropic missing api key,
        # but the routing should be tested
        pass

    # Directly test the routing by setting a provider not handled
    client.settings.provider = "invalid"
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("test")


@pytest.mark.asyncio
async def test_llm_client_ollama_uses_base_url():
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content="ollama response"))]
    ))

    with patch("furrow.llm.AsyncOpenAI", return_value=mock_client) as mock_cls:
        from furrow.config import Settings, Provider
        settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
        client = LLMClient(settings=settings)
        result = await client.complete("test", model="llama3")
        assert result == "ollama response"
        mock_cls.assert_called_once_with(base_url="http://localhost:11434", api_key="ollama")
