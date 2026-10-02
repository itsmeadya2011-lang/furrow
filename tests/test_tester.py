"""Tests for the Furrow tester agent."""

from __future__ import annotations

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import Settings


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self._response = response
        self.complete_calls: list[str] = []
        self.settings = Settings()

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.complete_calls.append(prompt)
        return self._response


@pytest.mark.asyncio
async def test_tester_passed():
    response = '{"passed": true, "summary": "All tests passed", "failures": []}'
    client = FakeLLMClient(response)
    tester = TesterAgent(client=client)  # type: ignore[arg-type]

    async def fake_run_tests() -> str:
        return "All tests passed.\n"

    tester._run_tests = fake_run_tests  # type: ignore[method-assign]

    result = await tester.run("do something", [])
    assert result.passed is True
    assert result.summary == "All tests passed"
    assert result.failures == []


@pytest.mark.asyncio
async def test_tester_failed_with_failures():
    response = '{"passed": false, "summary": "3 tests failed", "failures": ["test_a failed", "test_b failed"]}'
    client = FakeLLMClient(response)
    tester = TesterAgent(client=client)  # type: ignore[arg-type]

    async def fake_run_tests() -> str:
        return "3 tests failed.\n"

    tester._run_tests = fake_run_tests  # type: ignore[method-assign]

    result = await tester.run("do something", [])
    assert result.passed is False
    assert len(result.failures) == 2


@pytest.mark.asyncio
async def test_tester_invalid_json_fallback():
    response = "passed: yes, everything looks good"
    client = FakeLLMClient(response)
    tester = TesterAgent(client=client)  # type: ignore[arg-type]

    async def fake_run_tests() -> str:
        return "All good.\n"

    tester._run_tests = fake_run_tests  # type: ignore[method-assign]

    result = await tester.run("do something", [])
    assert result.passed is True
    assert "looks good" in result.summary


@pytest.mark.asyncio
async def test_tester_prompt_includes_goal():
    response = '{"passed": true, "summary": "ok", "failures": []}'
    client = FakeLLMClient(response)
    tester = TesterAgent(client=client)  # type: ignore[arg-type]

    async def fake_run_tests() -> str:
        return "Tests passed.\n"

    tester._run_tests = fake_run_tests  # type: ignore[method-assign]

    await tester.run("My specific goal", [])
    prompt = client.complete_calls[0]
    assert "My specific goal" in prompt


@pytest.mark.asyncio
async def test_tester_no_test_runner_found():
    """When no test runner is available, tester should return passed=True with message."""
    response = '{"passed": true, "summary": "No tests to run", "failures": []}'
    client = FakeLLMClient(response)
    tester = TesterAgent(client=client)  # type: ignore[arg-type]

    # Mock _run_tests to return no runner found
    async def fake_run_tests() -> str:
        return "No test runner found."

    tester._run_tests = fake_run_tests  # type: ignore[method-assign]

    result = await tester.run("some goal", [])
    assert result.passed is True
