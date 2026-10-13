"""Tests for the TesterAgent, including test runner detection and error handling."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import TaskModel, TestResult


class MockLLMClient:
    def __init__(self, response='{"passed": true, "summary": "ok", "failures": []}'):
        self.settings = MagicMock()
        self.settings.tester_model = "test-model"
        self._response = response

    async def complete(self, prompt, system="", model=None):
        return self._response


@pytest.mark.asyncio
async def test_tester_pass_result():
    """Tester should parse passing test results."""
    response = json.dumps({
        "passed": True, "summary": "All 10 tests passed", "failures": []
    })
    client = MockLLMClient(response)
    tester = TesterAgent(client=client)

    with patch.object(tester, "_run_tests", return_value="all passed"):
        result = await tester.run("test goal", [])
    assert result.passed is True
    assert result.failures == []
    assert "10" in result.summary


@pytest.mark.asyncio
async def test_tester_fail_result():
    """Tester should parse failing test results."""
    response = json.dumps({
        "passed": False,
        "summary": "2 failures",
        "failures": ["test_auth.py::test_login failed", "test_auth.py::test_logout failed"],
    })
    client = MockLLMClient(response)
    tester = TesterAgent(client=client)

    with patch.object(tester, "_run_tests", return_value="2 failures"):
        result = await tester.run("test goal", [])
    assert result.passed is False
    assert len(result.failures) == 2


@pytest.mark.asyncio
async def test_tester_invalid_json_fallback_pass():
    """Tester should fall back to keyword matching for invalid JSON (pass)."""
    client = MockLLMClient("The tests passed successfully with 5 cases.")
    tester = TesterAgent(client=client)

    with patch.object(tester, "_run_tests", return_value="some output"):
        result = await tester.run("test goal", [])
    assert result.passed is True  # "passed" in response.lower()


@pytest.mark.asyncio
async def test_tester_invalid_json_fallback_fail():
    """Tester should detect failure when JSON is invalid and keyword is 'failed'."""
    client = MockLLMClient("The tests failed due to errors.")
    tester = TesterAgent(client=client)

    with patch.object(tester, "_run_tests", return_value="some output"):
        result = await tester.run("test goal", [])
    assert result.passed is False


@pytest.mark.asyncio
async def test_tester_no_test_runner():
    """Tester should return 'No test runner found.' when no runners are available."""
    client = MockLLMClient()
    tester = TesterAgent(client=client)

    with patch.object(tester, "_detect_test_commands", return_value=[]):
        result = await tester._run_tests()
        assert "No test runner found" in result


@pytest.mark.asyncio
async def test_tester_file_not_found_skips_to_next():
    """Tester should skip commands not found and try the next."""
    client = MockLLMClient()
    tester = TesterAgent(client=client)

    with patch.object(
        tester, "_detect_test_commands",
        return_value=[["pytest", "-q"], ["python", "-m", "pytest", "-q"]],
    ):
        call_count = 0

        async def mock_exec(*cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise FileNotFoundError("pytest not found")
            mock_proc = MagicMock()
            mock_proc.communicate = AsyncMock(return_value=(b"test output", b""))
            mock_proc.returncode = 0
            return mock_proc

        with patch("furrow.agents.tester.asyncio.create_subprocess_exec", side_effect=mock_exec):
            result = await tester._run_tests()
            assert "test output" in result


@pytest.mark.asyncio
async def test_tester_includes_exit_code():
    """Test output should include the exit code."""
    client = MockLLMClient()
    tester = TesterAgent(client=client)

    with patch.object(tester, "_detect_test_commands", return_value=[["pytest", "-q"]]):
        mock_proc = MagicMock()
        mock_proc.communicate = AsyncMock(return_value=(b"all good", b""))
        mock_proc.returncode = 0

        with patch("furrow.agents.tester.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await tester._run_tests()
            assert "exit code: 0" in result


@pytest.mark.asyncio
async def test_tester_includes_exit_code_failure():
    """Test output should reflect non-zero exit codes."""
    client = MockLLMClient()
    tester = TesterAgent(client=client)

    with patch.object(tester, "_detect_test_commands", return_value=[["pytest", "-q"]]):
        mock_proc = MagicMock()
        mock_proc.communicate = AsyncMock(return_value=(b"FAILED test_foo", b""))
        mock_proc.returncode = 1

        with patch("furrow.agents.tester.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await tester._run_tests()
            assert "exit code: 1" in result
            assert "FAILED" in result


@pytest.mark.asyncio
async def test_tester_runs_llm_on_output():
    """Tester should pass test output to the LLM for analysis."""
    response = json.dumps({"passed": True, "summary": "ok", "failures": []})
    client = MockLLMClient(response)
    tester = TesterAgent(client=client)

    with patch.object(tester, "_run_tests", return_value="pytest output here"):
        with patch.object(client, "complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = response
            await tester.run("goal", [])
            mock_complete.assert_called_once()
            prompt = mock_complete.call_args[0][0]
            assert "pytest output here" in prompt
            assert "goal" in prompt


@pytest.mark.asyncio
async def test_tester_timeout_returns_message():
    """Tester should return a timeout message when tests exceed the timeout."""
    client = MockLLMClient()
    tester = TesterAgent(client=client)
    tester._timeout = 1

    async def mock_exec(*cmd, **kwargs):
        mock_proc = MagicMock()

        async def mock_communicate():
            await asyncio.sleep(10)  # Longer than timeout
            return (b"", b"")

        mock_proc.communicate = mock_communicate
        mock_proc.returncode = 0
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()  # Must be a coroutine
        mock_proc.wait.return_value = None
        return mock_proc

    with patch.object(tester, "_detect_test_commands", return_value=[["pytest", "-q"]]):
        with patch("furrow.agents.tester.asyncio.create_subprocess_exec", side_effect=mock_exec):
            result = await tester._run_tests()
            assert "timed out" in result.lower()
