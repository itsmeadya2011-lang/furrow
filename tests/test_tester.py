"""Tests for the TesterAgent."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import TaskModel, TestResult


@pytest.fixture
def tasks():
    return [TaskModel(id="1", description="Implement feature")]


@pytest.mark.asyncio
async def test_tester_parse_valid_result(mock_llm_client: MagicMock, tasks):
    """Tester returns a TestResult parsed from LLM JSON."""
    result_json = json.dumps({
        "passed": True,
        "summary": "All 10 tests passed",
        "failures": [],
    })
    mock_llm_client.complete = AsyncMock(return_value=result_json)

    # Mock _run_tests to avoid actually running tests
    tester = TesterAgent(client=mock_llm_client)
    tester._run_tests = AsyncMock(return_value="10 passed")

    result = await tester.run("Some goal", tasks)

    assert isinstance(result, TestResult)
    assert result.passed is True
    assert result.summary == "All 10 tests passed"
    assert result.failures == []


@pytest.mark.asyncio
async def test_tester_parse_failure_result(mock_llm_client: MagicMock, tasks):
    """Tester returns a TestResult with failures parsed from LLM JSON."""
    result_json = json.dumps({
        "passed": False,
        "summary": "2 tests failed",
        "failures": ["test_auth.py::test_login failed: assertion", "test_auth.py::test_logout failed: timeout"],
    })
    mock_llm_client.complete = AsyncMock(return_value=result_json)

    tester = TesterAgent(client=mock_llm_client)
    tester._run_tests = AsyncMock(return_value="2 failed")

    result = await tester.run("Some goal", tasks)

    assert result.passed is False
    assert len(result.failures) == 2


@pytest.mark.asyncio
async def test_tester_fallback_on_non_json(mock_llm_client: MagicMock, tasks):
    """When LLM returns non-JSON, tester falls back to keyword heuristic."""
    mock_llm_client.complete = AsyncMock(return_value="tests passed")

    tester = TesterAgent(client=mock_llm_client)
    tester._run_tests = AsyncMock(return_value="10 passed")

    result = await tester.run("Some goal", tasks)

    assert result.passed is True
    assert result.summary == "tests passed"


@pytest.mark.asyncio
async def test_tester_fallback_failed_keyword(mock_llm_client: MagicMock, tasks):
    """When LLM returns non-JSON with 'failed', tester heuristic returns False."""
    mock_llm_client.complete = AsyncMock(return_value="tests failed: 1 error")

    tester = TesterAgent(client=mock_llm_client)
    tester._run_tests = AsyncMock(return_value="1 failed")

    result = await tester.run("Some goal", tasks)

    assert result.passed is False


@pytest.mark.asyncio
async def test_tester_handles_runner_exception(mock_llm_client: MagicMock, tasks):
    """When _run_tests raises, tester returns a failing result without calling LLM."""
    mock_llm_client.complete = AsyncMock(return_value="should not be called")

    tester = TesterAgent(client=mock_llm_client)
    tester._run_tests = AsyncMock(side_effect=RuntimeError("Test runner crashed"))

    result = await tester.run("Some goal", tasks)

    assert result.passed is False
    assert "Test runner crashed" in result.summary
    mock_llm_client.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_tester_no_runner_found(mock_llm_client: MagicMock, tasks):
    """When no test runner is found, _run_tests returns a message string."""
    import shutil

    # Mock shutil.which to return None for all executables
    original_which = shutil.which
    shutil.which = MagicMock(return_value=None)
    try:
        tester = TesterAgent(client=mock_llm_client)
        output = await tester._run_tests()
        assert "No test runner found" in output
    finally:
        shutil.which = original_which  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_tester_uses_tester_model(mock_llm_client: MagicMock, tasks):
    """Tester uses the tester_model from settings."""
    result_json = json.dumps({"passed": True, "summary": "ok", "failures": []})
    mock_llm_client.complete = AsyncMock(return_value=result_json)

    tester = TesterAgent(client=mock_llm_client)
    tester._run_tests = AsyncMock(return_value="passed")

    await tester.run("Goal", tasks)

    call_kwargs = mock_llm_client.complete.call_args.kwargs
    assert call_kwargs["model"] == mock_llm_client.settings.tester_model


@pytest.mark.asyncio
async def test_tester_prompt_includes_goal_and_output(mock_llm_client: MagicMock, tasks):
    """Tester prompt includes the goal and test output."""
    result_json = json.dumps({"passed": True, "summary": "ok", "failures": []})
    mock_llm_client.complete = AsyncMock(return_value=result_json)

    tester = TesterAgent(client=mock_llm_client)
    tester._run_tests = AsyncMock(return_value="test output here")

    await tester.run("My goal", tasks)

    call_args = mock_llm_client.complete.call_args
    prompt = call_args.args[0]
    assert "My goal" in prompt
    assert "test output here" in prompt
