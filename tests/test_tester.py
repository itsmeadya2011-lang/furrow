import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.config import TestResult, Settings
from furrow.agents.tester import TesterAgent


def test_test_result_defaults():
    t = TestResult(passed=True, summary="ok")
    assert t.failures == []


@pytest.mark.asyncio
async def test_tester_run_mocked(monkeypatch):
    response = json.dumps({"passed": True, "summary": "all good", "failures": []})

    client = MagicMock()
    client.complete = AsyncMock(return_value=response)
    client.settings = Settings()

    agent = TesterAgent(client=client)
    monkeypatch.setattr(agent, "_run_tests", AsyncMock(return_value="fake output"))

    result = await agent.run("some goal", [])
    assert isinstance(result, TestResult)
    assert result.passed is True
    assert result.summary == "all good"
    assert result.failures == []


@pytest.mark.asyncio
async def test_tester_parsing_from_malformed(monkeypatch):
    client = MagicMock()
    client.complete = AsyncMock(return_value="not json at all")
    client.settings = Settings()

    agent = TesterAgent(client=client)
    monkeypatch.setattr(agent, "_run_tests", AsyncMock(return_value="fake output"))

    result = await agent.run("some goal", [])
    assert isinstance(result, TestResult)
    assert result.passed is False
    assert result.summary == "not json at all"
    assert result.failures == []
