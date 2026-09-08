import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from furrow.agents.tester import TesterAgent
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient


def test_tester_agent_init():
    client = LLMClient()
    agent = TesterAgent(client=client)
    assert agent.client is client


def test_tester_agent_init_defaults():
    agent = TesterAgent()
    assert agent.client is not None


@pytest.mark.asyncio
async def test_run_tests_timeout_zombie_fix():
    settings = Settings()
    client = LLMClient(settings=settings)
    agent = TesterAgent(client=client, settings=settings)

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
    mock_proc.kill = AsyncMock()
    mock_proc.wait = AsyncMock()

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await agent._run_tests()

    mock_proc.kill.assert_called_once()
    mock_proc.wait.assert_called_once()
    assert result == "No test runner found."


@pytest.mark.asyncio
async def test_run_parses_json_response():
    settings = Settings()
    client = LLMClient(settings=settings)
    agent = TesterAgent(client=client, settings=settings)

    response = json.dumps({"passed": True, "summary": "all good", "failures": []})
    with patch.object(client, "complete", new_callable=AsyncMock, return_value=response):
        result = await agent.run("goal", [])

    assert result.passed is True
    assert result.summary == "all good"
