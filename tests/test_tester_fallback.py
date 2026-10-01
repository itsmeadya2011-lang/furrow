"""Tests for the tester agent's JSON parse fallback."""
from __future__ import annotations

import json

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import Settings, TestResult


class _FakeLLMClient:
    """A fake LLMClient that returns canned responses."""

    def __init__(self, response: str):
        self._response = response
        self.settings = Settings()

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        return self._response


@pytest.mark.asyncio
async def test_tester_valid_json():
    """Tester parses valid JSON from LLM."""
    result = {"passed": True, "summary": "All tests passed", "failures": []}
    client = _FakeLLMClient(response=json.dumps(result))
    tester = TesterAgent(client=client)
    # Bypass _run_tests by temporarily overriding it
    tester._run_tests = lambda: _async_value("test output")
    tr = await tester.run("goal", [])
    assert isinstance(tr, TestResult)
    assert tr.passed is True
    assert tr.summary == "All tests passed"


@pytest.mark.asyncio
async def test_tester_fallback_all_passed():
    """Tester fallback handles 'all tests passed' text."""
    client = _FakeLLMClient(response="All tests passed successfully!")
    tester = TesterAgent(client=client)
    tester._run_tests = lambda: _async_value("test output")
    tr = await tester.run("goal", [])
    assert tr.passed is True


@pytest.mark.asyncio
async def test_tester_fallback_tests_failed():
    """Tester fallback handles 'tests failed' text."""
    client = _FakeLLMClient(response="tests failed: assertion error in test_foo")
    tester = TesterAgent(client=client)
    tester._run_tests = lambda: _async_value("test output")
    tr = await tester.run("goal", [])
    assert tr.passed is False


@pytest.mark.asyncio
async def test_tester_fallback_json_with_passed_key():
    """Tester fallback extracts passed value from JSON-like text."""
    client = _FakeLLMClient(response='Result: {"passed": false, "summary": "bad"}')
    tester = TesterAgent(client=client)
    tester._run_tests = lambda: _async_value("test output")
    tr = await tester.run("goal", [])
    assert tr.passed is False


async def _async_value(value: str) -> str:
    return value
