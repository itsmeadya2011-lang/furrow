import json
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_provider_enum():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


def test_task_model_defaults():
    t = TaskModel(id="1", description="test")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"
    assert s.ollama_base_url == "http://localhost:11434"


def test_llm_client_anthropic_missing_key():
    s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
            LLMClient(settings=s)


def test_llm_client_openai_missing_key():
    s = Settings(provider=Provider.OPENAI, openai_api_key=None)
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
            LLMClient(settings=s)


def test_llm_client_ollama_no_api_key_needed():
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=s)
    assert client.settings.provider == Provider.OLLAMA


@pytest.mark.asyncio
async def test_llm_client_complete_openai():
    s = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=s)
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello"
    client._openai = AsyncMock()
    client._openai.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await client.complete("Hi", model="gpt-4")
    assert result == "Hello"
    client._openai.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_llm_client_complete_anthropic():
    s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
    client = LLMClient(settings=s)
    mock_content = MagicMock()
    mock_content.text = "Hello"
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    client._anthropic = AsyncMock()
    client._anthropic.messages.create = AsyncMock(return_value=mock_response)

    result = await client.complete("Hi", model="claude-3-5-sonnet-20241022")
    assert result == "Hello"
    client._anthropic.messages.create.assert_called_once()


@pytest.mark.asyncio
async def test_planner_agent_parses_json():
    s = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=s)
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps({
        "tasks": [{"id": "1", "description": "do thing", "files": [], "dependencies": []}],
        "rationale": "ok"
    })
    client._openai = AsyncMock()
    client._openai.chat.completions.create = AsyncMock(return_value=mock_response)

    planner = PlannerAgent(client=client)
    plan = await planner.plan("build a thing")
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "do thing"


@pytest.mark.asyncio
async def test_planner_agent_invalid_json_raises():
    s = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=s)
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "not json"
    client._openai = AsyncMock()
    client._openai.chat.completions.create = AsyncMock(return_value=mock_response)

    planner = PlannerAgent(client=client)
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("build a thing")


def test_orchestrator_get_tasks_empty():
    o = Orchestrator(goal="test")
    assert o._get_tasks() == []


def test_orchestrator_is_done_no_tasks():
    o = Orchestrator(goal="test")
    assert o._is_done() is True


def test_orchestrator_is_done_with_completed_tasks():
    o = Orchestrator(goal="test")
    o.plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="completed"),
        ],
        rationale="ok",
    )
    assert o._is_done() is True


def test_orchestrator_is_done_with_failed_tasks():
    o = Orchestrator(goal="test")
    o.plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="failed"),
        ],
        rationale="ok",
    )
    assert o._is_done() is False


def test_orchestrator_is_done_with_pending_tasks():
    o = Orchestrator(goal="test")
    o.plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="pending"),
        ],
        rationale="ok",
    )
    assert o._is_done() is False


@pytest.mark.asyncio
async def test_tester_agent_no_test_runner():
    s = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=s)

    with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
        tester = TesterAgent(client=client)
        output = await tester._run_tests()
        assert "No test runner found" in output


@pytest.mark.asyncio
async def test_tester_agent_successful_run():
    s = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=s)

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"ok\n", b""))
    mock_proc.returncode = 0
    mock_proc.kill = MagicMock()

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        tester = TesterAgent(client=client)
        output = await tester._run_tests()
        assert "ok" in output


@pytest.mark.asyncio
async def test_tester_agent_timeout():
    s = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
    client = LLMClient(settings=s)

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
    mock_proc.kill = MagicMock()

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        tester = TesterAgent(client=client)
        output = await tester._run_tests()
        assert "timed out" in output
