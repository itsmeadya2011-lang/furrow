"""Tests for the TesterAgent."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import Settings, TaskModel


def make_mock_client(test_response=None, complete_mock=None):
    """Create a mock LLMClient."""
    client = MagicMock()
    client.settings = Settings()
    if complete_mock:
        client.complete = complete_mock
    else:
        client.complete = AsyncMock(return_value=test_response or '{"passed": true, "summary": "All good", "failures": []}')
    return client


@pytest.fixture(autouse=True)
def _patch_create_subprocess_exec():
    """Avoid spawning a real test runner (which would recurse into pytest)."""
    async def _raise_no_such_file(*args, **kwargs):
        raise FileNotFoundError("no test runner available")

    with patch("furrow.agents.tester.asyncio.create_subprocess_exec", side_effect=_raise_no_such_file):
        yield


async def test_tester_success():
    """Tester parses passing test result."""
    client = make_mock_client('{"passed": true, "summary": "All tests passed", "failures": []}')
    tester = TesterAgent(client=client)
    tasks = [TaskModel(id="1", description="task")]
    result = await tester.run("Build feature", tasks)
    assert result.passed is True
    assert result.summary == "All tests passed"
    assert result.failures == []


async def test_tester_failure():
    """Tester parses failing test result."""
    client = make_mock_client('{"passed": false, "summary": "2 failures", "failures": ["test_a", "test_b"]}')
    tester = TesterAgent(client=client)
    tasks = [TaskModel(id="1", description="task")]
    result = await tester.run("Fix things", tasks)
    assert result.passed is False
    assert result.summary == "2 failures"
    assert len(result.failures) == 2


async def test_tester_fallback_on_invalid_json():
    """Tester falls back to heuristic when JSON parse fails."""
    client = make_mock_client("Tests passed. Everything is good.")
    tester = TesterAgent(client=client)
    tasks = [TaskModel(id="1", description="task")]
    result = await tester.run("Goal", tasks)
    assert result.passed is True
    assert "passed" in result.summary.lower()


async def test_tester_fallback_json_failure():
    """Tester fallback correctly detects failure."""
    client = make_mock_client("Tests failed. Some error occurred.")
    tester = TesterAgent(client=client)
    tasks = [TaskModel(id="1", description="task")]
    result = await tester.run("Goal", tasks)
    assert result.passed is False


async def test_tester_uses_correct_model():
    """Tester uses the tester_model from settings."""
    client = make_mock_client('{"passed": true, "summary": "ok", "failures": []}')
    tester = TesterAgent(client=client)
    tasks = [TaskModel(id="1", description="task")]
    await tester.run("Goal", tasks)
    call_args = client.complete.call_args
    assert call_args.kwargs.get("model") == client.settings.tester_model


async def test_tester_no_test_runner():
    """Tester handles case where no test runner is found."""
    # Mock _run_tests to return the no-runner message without actually running
    async def mock_run_tests():
        return "No test runner found."

    client = make_mock_client('{"passed": true, "summary": "No test runner, but LLM reports all good", "failures": []}')
    tester = TesterAgent(client=client)
    tester._run_tests = mock_run_tests
    tasks = [TaskModel(id="1", description="task")]
    result = await tester.run("Goal", tasks)
    assert result.passed is True  # LLM still says passed even if no runner


async def test_tester_runner_not_found():
    """Tester's _run_tests returns message when no runner is available."""
    from furrow.agents.tester import TesterAgent
    tester = TesterAgent(client=MagicMock())
    tester.client = MagicMock()

    # Patch create_subprocess_exec to raise FileNotFoundError for all commands
    async def mock_exec(*args, **kwargs):
        raise FileNotFoundError(f"No such file: {args[0]}")

    with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
        result = await tester._run_tests()
        assert "No test runner found" in result
