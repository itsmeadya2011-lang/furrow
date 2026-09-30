from types import SimpleNamespace

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import TestResult


class FakeClient:
    def __init__(self, response):
        self._response = response

    @property
    def settings(self):
        return SimpleNamespace(tester_model="x")

    async def complete(self, prompt, system="", model=None):
        return self._response


@pytest.mark.asyncio
async def test_run_tests_returns_no_runner_when_none():
    agent = TesterAgent(client=FakeClient('{"passed": true, "summary": "ok", "failures": []}'))
    agent._run_tests = lambda: _no_runner_return()
    result = await agent.run(goal="g", tasks=[])
    assert result.passed is True
    assert result.summary == "ok"
    assert result.failures == []


async def _no_runner_return() -> str:
    return "No test runner found."


@pytest.mark.asyncio
async def test_run_parses_passed_json():
    agent = TesterAgent(
        client=FakeClient('{"passed": true, "summary": "all good", "failures": []}')
    )
    agent._run_tests = lambda: _ok_return()
    result = await agent.run(goal="g", tasks=[])
    assert result == TestResult(passed=True, summary="all good", failures=[])


async def _ok_return() -> str:
    return "ok"


@pytest.mark.asyncio
async def test_run_falls_back_to_keyword():
    agent = TesterAgent(client=FakeClient("tests passed!"))
    agent._run_tests = lambda: _ok_return()
    result = await agent.run(goal="g", tasks=[])
    assert result.passed is True
    assert result.summary == "tests passed!"
    assert result.failures == []


@pytest.mark.asyncio
async def test_run_failure_json():
    agent = TesterAgent(
        client=FakeClient('{"passed": false, "summary": "boom", "failures": ["x"]}')
    )
    agent._run_tests = lambda: _ok_return()
    result = await agent.run(goal="g", tasks=[])
    assert result.passed is False
    assert result.summary == "boom"
    assert result.failures == ["x"]


@pytest.mark.asyncio
async def test_run_tests_skips_missing_runner(monkeypatch):
    async def raise_not_found(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr("asyncio.create_subprocess_exec", raise_not_found)
    agent = TesterAgent(client=FakeClient("ignored"))
    result = await agent._run_tests()
    assert result == "No test runner found."
