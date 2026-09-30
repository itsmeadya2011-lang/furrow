"""Tests for furrow.agents.tester."""
from __future__ import annotations

import json

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import TaskModel
from furrow.llm import LLMClient


class _FakeClient:
    def __init__(self, response: str) -> None:
        self._response = response
        self.settings = LLMClient().settings
        self.complete_calls: list[tuple[str, str, str | None]] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.complete_calls.append((prompt, system, model))
        return self._response


@pytest.mark.asyncio
async def test_tester_parses_json_result(monkeypatch: pytest.MonkeyPatch):
    """When the LLM returns valid JSON, TestResult should be constructed from it."""
    fake_json = json.dumps({
        "passed": True,
        "summary": "All 5 tests passed",
        "failures": [],
    })
    client = _FakeClient(fake_json)
    tester = TesterAgent(client=client)
    # Stub out subprocess test execution so we don't run real tests.
    monkeypatch.setattr(tester, "_run_tests", _return_test_output("dummy output"))
    result = await tester.run("some goal", [TaskModel(id="1", description="x")])
    assert result.passed is True
    assert result.summary == "All 5 tests passed"
    assert result.failures == []


@pytest.mark.asyncio
async def test_tester_fallback_on_invalid_json(monkeypatch: pytest.MonkeyPatch):
    """When JSON parsing fails, fall back to keyword detection."""
    client = _FakeClient("tests all passed, 10 passed")
    tester = TesterAgent(client=client)
    monkeypatch.setattr(tester, "_run_tests", _return_test_output("ok"))
    result = await tester.run("goal", [])
    assert result.passed is True
    assert "passed" in result.summary


@pytest.mark.asyncio
async def test_tester_failed_tests(monkeypatch: pytest.MonkeyPatch):
    """When tests fail, the tester reports failures."""
    fake_json = json.dumps({
        "passed": False,
        "summary": "2 tests failed",
        "failures": ["test_auth failed: assertion", "test_db failed: timeout"],
    })
    client = _FakeClient(fake_json)
    tester = TesterAgent(client=client)
    monkeypatch.setattr(tester, "_run_tests", _return_test_output("failure output"))
    result = await tester.run("goal", [])
    assert result.passed is False
    assert len(result.failures) == 2


@pytest.mark.asyncio
async def test_tester_captured_error(monkeypatch: pytest.MonkeyPatch):
    """When _run_tests raises, return a failing TestResult."""
    client = _FakeClient("")
    tester = TesterAgent(client=client)

    async def _raise() -> str:
        raise RuntimeError("pytest not found")

    monkeypatch.setattr(tester, "_run_tests", _raise)
    result = await tester.run("goal", [])
    assert result.passed is False
    assert "pytest not found" in result.summary


@pytest.mark.asyncio
async def test_tester_uses_tester_model(monkeypatch: pytest.MonkeyPatch):
    client = _FakeClient(json.dumps({"passed": True, "summary": "ok", "failures": []}))
    tester = TesterAgent(client=client)
    monkeypatch.setattr(tester, "_run_tests", _return_test_output("ok"))
    await tester.run("goal", [])
    assert client.complete_calls[0][2] == client.settings.tester_model


def _return_test_output(text: str):
    """Return a coroutine function that yields *text*."""

    async def _fake() -> str:
        return text

    return _fake
