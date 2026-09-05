import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Provider, Settings, TaskModel, TestResult


@pytest.mark.asyncio
async def test_planner_agent_formats_prompt_with_project_context():
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value='{"tasks": [], "rationale": "ok"}')
    mock_client.settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test")

    agent = PlannerAgent(client=mock_client)
    plan = await agent.plan("build auth", project_context="src/auth.py\nsrc/main.py")

    call_args = mock_client.complete.call_args
    prompt = call_args[0][0]
    assert "src/auth.py" in prompt
    assert "src/main.py" in prompt
    assert "build auth" in prompt


@pytest.mark.asyncio
async def test_planner_agent_raises_on_invalid_json():
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value="not valid json")
    mock_client.settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test")

    agent = PlannerAgent(client=mock_client)
    with pytest.raises(ValueError, match="Failed to parse plan from LLM"):
        await agent.plan("build auth")


@pytest.mark.asyncio
async def test_worker_agent_formats_prompt_with_project_context():
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value="done")
    mock_client.settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test")
    mock_client.settings.workspace = MagicMock()

    task = TaskModel(id="1", description="do thing", files=["src/a.py"])
    agent = WorkerAgent(task=task, client=mock_client)

    with patch("os.walk", return_value=[("/workspace", [], ["a.py"])]):
        result = await agent.run(project_context="src/a.py")

    call_args = mock_client.complete.call_args
    prompt = call_args[0][0]
    assert "do thing" in prompt
    assert "src/a.py" in prompt


@pytest.mark.asyncio
async def test_tester_run_tests_uses_primary_test_command():
    settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    mock_client = MagicMock()
    mock_client.settings = settings
    mock_client.settings.get_test_command.return_value = ["pytest", "-q"]

    agent = TesterAgent(client=mock_client)

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"passed", b""))

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        output = await agent._run_tests()

    assert "passed" in output


@pytest.mark.asyncio
async def test_tester_run_returns_passed_true_when_no_test_output():
    settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    mock_client = MagicMock()
    mock_client.settings = settings

    agent = TesterAgent(client=mock_client)
    with patch.object(agent, "_run_tests", new_callable=AsyncMock, return_value=""):
        result = await agent.run("build auth", [])

    assert result.passed is True
    assert "No tests found" in result.summary