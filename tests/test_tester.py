"""Tests for the TesterAgent — test result parsing and formatting."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import TaskModel


@pytest.fixture
def client():
    c = MagicMock()
    c.settings.tester_model = "model"
    c.complete = AsyncMock()
    return c


def test_parse_test_result_json():
    response = json.dumps({
        "passed": True,
        "summary": "All tests passed",
        "failures": [],
    })
    result = TesterAgent._parse_test_result(response)
    assert result.passed is True
    assert result.summary == "All tests passed"


def test_parse_test_result_markdown_fenced():
    response = '```json\n' + json.dumps({
        "passed": False,
        "summary": "2 tests failed",
        "failures": ["test_a", "test_b"],
    }) + '\n```'
    result = TesterAgent._parse_test_result(response)
    assert result.passed is False
    assert len(result.failures) == 2


def test_parse_test_result_fallback_heuristic():
    response = "All tests passed successfully!"
    result = TesterAgent._parse_test_result(response)
    assert result.passed is True


def test_parse_test_result_fallback_with_failures():
    response = "Some tests failed. 3 failures."
    result = TesterAgent._parse_test_result(response)
    assert result.passed is False
    assert result.failures == []


def test_format_tasks():
    tasks = [
        TaskModel(id="1", description="Create auth", status="completed"),
        TaskModel(id="2", description="Write tests", status="failed"),
    ]
    formatted = TesterAgent._format_tasks(tasks)
    assert "[1]" in formatted
    assert "Create auth" in formatted
    assert "[2]" in formatted
    assert "Write tests" in formatted


@pytest.mark.asyncio
async def test_tester_returns_failed_on_test_error(client):
    """If the test runner itself errors, return a failed TestResult."""
    client.complete = AsyncMock()  # not called if _run_tests fails
    tester = TesterAgent(client=client)
    tasks = [TaskModel(id="1", description="x")]

    # Force _run_tests to raise.
    import asyncio as _aio
    tester._run_tests = AsyncMock(side_effect=_aio.TimeoutError("runner crashed"))
    result = await tester.run("goal", tasks)
    assert result.passed is False
    assert "runner crashed" in result.summary
