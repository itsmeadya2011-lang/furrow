from __future__ import annotations

import asyncio
from unittest import mock

from furrow.agents.tester import TesterAgent
from furrow.config import TaskModel, TestResult


def _task() -> TaskModel:
    return TaskModel(id="1", description="do the thing")


def _make_client(
    return_value: str = '{"passed": true, "summary": "ok", "failures": []}',
) -> mock.MagicMock:
    client = mock.MagicMock()
    client.settings = mock.MagicMock(tester_model="test-model")
    client.complete = mock.AsyncMock(return_value=return_value)
    return client


async def test_run_passes_when_returncode_zero_and_llm_passes():
    client = _make_client('{"passed": true, "summary": "ok", "failures": []}')
    agent = TesterAgent(client=client)
    agent._run_tests = mock.AsyncMock(return_value=("all good", 0))

    result = await agent.run(goal="do the thing", tasks=[_task()])

    assert result.passed is True
    assert result.summary == "ok"
    assert result.failures == []


async def test_run_fails_when_returncode_nonzero_even_if_llm_passes():
    client = _make_client('{"passed": true, "summary": "ok", "failures": []}')
    agent = TesterAgent(client=client)
    agent._run_tests = mock.AsyncMock(return_value=("something broke", 1))

    result = await agent.run(goal="do the thing", tasks=[_task()])

    assert result.passed is False
    assert any("non-zero" in f.lower() for f in result.failures)


async def test_run_handles_unexpected_run_tests_error():
    client = _make_client('{"passed": true, "summary": "ok", "failures": []}')
    agent = TesterAgent(client=client)
    agent._run_tests = mock.AsyncMock(side_effect=RuntimeError("kaboom"))

    result = await agent.run(goal="do the thing", tasks=[_task()])

    assert result.passed is False
    assert result.summary == "kaboom"
    assert result.failures == ["kaboom"]


async def test_run_falls_back_when_llm_returns_non_json():
    client = _make_client(return_value="looks like tests passed but no json")
    agent = TesterAgent(client=client)
    agent._run_tests = mock.AsyncMock(return_value=("output", 0))

    result = await agent.run(goal="do the thing", tasks=[_task()])

    assert isinstance(result, TestResult)
    assert result.passed is True
    assert result.summary == "looks like tests passed but no json"
    assert result.failures == []


async def test_run_tests_returns_output_and_returncode(monkeypatch):
    class FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"tests ok", b"")

        def kill(self) -> None:
            pass

    async def fake_exec(*_args, **_kwargs) -> FakeProc:
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    agent = TesterAgent(client=_make_client())

    output, returncode = await agent._run_tests()

    assert output == "tests ok"
    assert returncode == 0


async def test_run_tests_no_runner_found(monkeypatch):
    def raise_fnf(*_args, **_kwargs):
        raise FileNotFoundError("command not found")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", raise_fnf)
    agent = TesterAgent(client=_make_client())

    output, returncode = await agent._run_tests()

    assert output == "No test runner found."
    assert returncode == -1


async def test_run_tests_nonzero_returncode_is_captured(monkeypatch):
    class FakeProc:
        returncode = 5

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"fail", b"error")

        def kill(self) -> None:
            pass

    async def fake_exec(*_args, **_kwargs) -> FakeProc:
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    agent = TesterAgent(client=_make_client())

    output, returncode = await agent._run_tests()

    assert output == "failerror"
    assert returncode == 5
