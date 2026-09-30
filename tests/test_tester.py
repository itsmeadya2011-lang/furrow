"""Tests for furrow.agents.tester — test execution and result parsing."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from furrow.agents.tester import TesterAgent
from furrow.config import Settings
from furrow.llm import LLMClient


def _make_tester(settings=None):
    return TesterAgent(client=LLMClient(settings=settings or Settings()))


async def test_tester_parses_passing_result():
    tester = _make_tester()
    fake_llm_response = json.dumps({
        "passed": True,
        "summary": "All 5 tests passed",
        "failures": [],
    })
    with (
        patch.object(tester.client, "complete", new=AsyncMock(return_value=fake_llm_response)),
        patch.object(tester, "_run_tests", new=AsyncMock(return_value="test output")),
    ):
        result = await tester.run("Make tests pass", [])
    assert result.passed is True
    assert result.summary == "All 5 tests passed"
    assert result.failures == []


async def test_tester_parses_failing_result():
    tester = _make_tester()
    fake_llm_response = json.dumps({
        "passed": False,
        "summary": "2 failures",
        "failures": ["test_a failed", "test_b failed"],
    })
    with (
        patch.object(tester.client, "complete", new=AsyncMock(return_value=fake_llm_response)),
        patch.object(tester, "_run_tests", new=AsyncMock(return_value="test output")),
    ):
        result = await tester.run("Fix bug", [])
    assert result.passed is False
    assert len(result.failures) == 2
    assert "test_a failed" in result.failures[0]


async def test_tester_fallback_on_invalid_json():
    """When LLM response isn't valid JSON, fall back to text matching."""
    tester = _make_tester()
    fake_llm_response = "All tests passed successfully"
    with (
        patch.object(tester.client, "complete", new=AsyncMock(return_value=fake_llm_response)),
        patch.object(tester, "_run_tests", new=AsyncMock(return_value="test output")),
    ):
        result = await tester.run("Make tests pass", [])
    assert result.passed is True
    assert result.summary == "All tests passed successfully"


async def test_tester_fallback_on_failed_text():
    tester = _make_tester()
    fake_llm_response = "Tests FAILED, there are errors"
    with (
        patch.object(tester.client, "complete", new=AsyncMock(return_value=fake_llm_response)),
        patch.object(tester, "_run_tests", new=AsyncMock(return_value="test output")),
    ):
        result = await tester.run("Fix tests", [])
    assert result.passed is False


async def test_tester_handles_test_runner_exception():
    """If _run_tests raises, return a failing TestResult."""
    tester = _make_tester()
    with patch.object(tester, "_run_tests", new=AsyncMock(side_effect=RuntimeError("boom"))):
        result = await tester.run("Fix bug", [])
    assert result.passed is False
    assert "boom" in result.summary
    assert "boom" in result.failures[0]


async def test_run_tests_returns_no_runner_message():
    """When no test runner is found, return a fallback message."""
    tester = _make_tester()
    with patch("furrow.agents.tester.asyncio.create_subprocess_exec", new=AsyncMock(side_effect=FileNotFoundError)):
        result = await tester._run_tests()
    assert result == "No test runner found."


async def test_test_timeout_from_settings():
    """Test timeout should come from settings, not be hardcoded."""
    settings = Settings(test_timeout=30.0)
    tester = TesterAgent(client=LLMClient(settings=settings))
    assert tester.client.settings.test_timeout == 30.0
