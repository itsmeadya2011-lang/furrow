from pathlib import Path
from types import SimpleNamespace

import pytest

from furrow.agents.tester import TesterAgent


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.settings = SimpleNamespace(tester_model="test-model")

    async def complete(self, prompt, system="", model=None):
        return self.response


async def test_run_tests_no_runner(monkeypatch, tmp_path):
    async def fake_exec(*args, **kwargs):
        raise FileNotFoundError("no runner")

    monkeypatch.setattr("furrow.agents.tester.asyncio.create_subprocess_exec", fake_exec)
    agent = TesterAgent(client=FakeClient("irrelevant"))
    result = await agent._run_tests()
    assert result == "No test runner found."


async def test_run_handles_no_runner_gracefully(monkeypatch, tmp_path):
    async def fake_exec(*args, **kwargs):
        raise FileNotFoundError("no runner")

    monkeypatch.setattr("furrow.agents.tester.asyncio.create_subprocess_exec", fake_exec)
    client = FakeClient('{"passed": false, "summary": "no runner found", "failures": []}')
    agent = TesterAgent(client=client)
    result = await agent.run("goal", [])
    assert result.passed is False
    assert result.summary == "no runner found"
    assert result.failures == []
