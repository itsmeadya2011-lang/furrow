import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from furrow.config import Plan, TaskModel, TestResult, Settings, Provider
from furrow.llm import LLMClient
from furrow.agents.planner import PlannerAgent
from furrow.agents.worker import WorkerAgent
from furrow.agents.tester import TesterAgent
from furrow.core.orchestrator import Orchestrator


def test_task_model_defaults():
    t = TaskModel(id="1", description="x")
    assert t.status == "pending"
    assert t.files == []
    assert t.dependencies == []


def test_plan_multiple_tasks():
    p = Plan(
        tasks=[
            TaskModel(id="1", description="a"),
            TaskModel(id="2", description="b"),
        ],
        rationale="do a and b",
    )
    assert p.rationale == "do a and b"
    assert len(p.tasks) == 2


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="fail", failures=["e1", "e2"])
    assert t.passed is False
    assert t.failures == ["e1", "e2"]


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("FURROW_MAX_CYCLES", "7")
    s = Settings()
    assert s.max_cycles == 7


async def test_llm_client_anthropic_routing():
    mock_anthropic = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="ok")]
    mock_anthropic.messages.create.return_value = mock_response

    with patch("furrow.llm.AsyncAnthropic", return_value=mock_anthropic):
        s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
        client = LLMClient(settings=s)
        result = await client.complete("hi")
        assert result == "ok"
        mock_anthropic.messages.create.assert_called_once()


async def test_llm_client_openai_routing():
    mock_openai = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
    mock_openai.chat.completions.create.return_value = mock_response

    with patch("furrow.llm.AsyncOpenAI", return_value=mock_openai):
        s = Settings(provider=Provider.OPENAI, openai_api_key="test-key")
        client = LLMClient(settings=s)
        result = await client.complete("hi")
        assert result == "ok"
        mock_openai.chat.completions.create.assert_called_once()


async def test_llm_client_ollama_routing(monkeypatch):
    import httpx

    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": "ok"}}
    mock_response.raise_for_status.return_value = None
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    with patch.object(httpx, "AsyncClient", return_value=mock_client):
        s = Settings(provider=Provider.OLLAMA)
        client = LLMClient(settings=s)
        result = await client.complete("hi")
        assert result == "ok"


async def test_planner_agent_parses_json():
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value='{"tasks": [], "rationale": "ok"}')

    agent = PlannerAgent(client=mock_client)
    plan = await agent.plan("g")
    assert isinstance(plan, Plan)
    assert plan.rationale == "ok"


async def test_worker_agent_prompt():
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value="done")

    agent = WorkerAgent(TaskModel(id="1", description="build X"), client=mock_client)
    await agent.run()

    prompt = mock_client.complete.call_args.args[0]
    assert "build X" in prompt


async def test_tester_truncation(monkeypatch):
    mock_client = MagicMock()
    captured_prompt = None

    async def capture_complete(prompt, **kwargs):
        nonlocal captured_prompt
        captured_prompt = prompt
        return '{"passed": true, "summary": "ok", "failures": []}'

    mock_client.complete = AsyncMock(side_effect=capture_complete)

    agent = TesterAgent(client=mock_client)
    long_output = "x" * 10000
    monkeypatch.setattr(agent, "_run_tests", AsyncMock(return_value=long_output))

    await agent.run("goal", [])

    test_output_start = captured_prompt.index("Test output:\n") + len("Test output:\n")
    test_output_in_prompt = captured_prompt[test_output_start:]
    assert len(test_output_in_prompt) <= 4020
    assert test_output_in_prompt.endswith("...[truncated]")


def test_orchestrator_is_done():
    orch = Orchestrator(goal="x")
    orch.all_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orch._is_done() is True

    orch.all_tasks[0].status = "pending"
    assert orch._is_done() is False

    orch.all_tasks[0].status = "completed"
    orch.all_tasks[1].status = "failed"
    assert orch._is_done() is False
