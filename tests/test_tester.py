from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import Settings, TaskModel, TestResult


class TestTesterAgent:
    """Tests for TesterAgent."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock LLM client."""
        client = MagicMock()
        client.complete = AsyncMock(return_value='{"passed": true, "summary": "All good", "failures": []}')
        client.settings = Settings()
        return client

    def test_find_test_command(self, mock_client: MagicMock) -> None:
        """_find_test_command returns a valid command or None."""
        tester = TesterAgent(client=mock_client)

        # Should return a command if pytest is available, or None
        result = tester._find_test_command()
        if shutil.which("pytest"):
            assert result is not None
            assert "pytest" in result[0]
        else:
            assert result is None

    def test_find_lint_command(self, mock_client: MagicMock) -> None:
        """_find_lint_command returns ruff command or None."""
        tester = TesterAgent(client=mock_client)

        result = tester._find_lint_command()
        if shutil.which("ruff"):
            assert result is not None
            assert result == ["ruff", "check", "furrow/", "tests/"]
        else:
            assert result is None

    def test_find_typecheck_command(self, mock_client: MagicMock) -> None:
        """_find_typecheck_command returns mypy command or None."""
        tester = TesterAgent(client=mock_client)

        result = tester._find_typecheck_command()
        if shutil.which("mypy"):
            assert result is not None
            assert result == ["mypy", "furrow/"]
        else:
            assert result is None

    @pytest.mark.asyncio
    async def test_run_command_none(self, mock_client: MagicMock) -> None:
        """_run_command(None, "name") returns None."""
        tester = TesterAgent(client=mock_client)

        result = await tester._run_command(None, "Tests")
        assert result is None

    @pytest.mark.asyncio
    async def test_run_command_timeout(self, mock_client: MagicMock) -> None:
        """_run_command returns timeout message for slow command."""
        tester = TesterAgent(client=mock_client)

        # Use a command that will hang (sleep for longer than timeout)
        # We'll mock the subprocess to simulate timeout
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
            mock_proc.kill = MagicMock()
            mock_proc.wait = AsyncMock()
            mock_subprocess.return_value = mock_proc

            result = await tester._run_command(["sleep", "200"], "Test")

            assert result is not None
            assert "timed out" in result.lower()
            assert "120 seconds" in result

    @pytest.mark.asyncio
    async def test_run_command_success(self, mock_client: MagicMock) -> None:
        """_run_command returns formatted output for successful command."""
        tester = TesterAgent(client=mock_client)

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"stdout output", b"stderr output"))
            mock_subprocess.return_value = mock_proc

            result = await tester._run_command(["echo", "hello"], "Test")

            assert result is not None
            assert "=== Test ===" in result
            assert "stdout output" in result
            assert "stderr output" in result

    @pytest.mark.asyncio
    async def test_run_command_launch_failed(self, mock_client: MagicMock) -> None:
        """_run_command returns error message when command launch fails."""
        tester = TesterAgent(client=mock_client)

        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError("No such file")):
            result = await tester._run_command(["nonexistent"], "Test")

            assert result is not None
            assert "=== Test ===" in result
            assert "Failed to launch command" in result

    @pytest.mark.asyncio
    async def test_run_all_checks_no_runner(self, mock_client: MagicMock) -> None:
        """_run_all_checks returns 'No test runner found.' when nothing available."""
        tester = TesterAgent(client=mock_client)

        # Mock all find methods to return None
        tester._find_test_command = MagicMock(return_value=None)
        tester._find_lint_command = MagicMock(return_value=None)
        tester._find_typecheck_command = MagicMock(return_value=None)

        result = await tester._run_all_checks()

        assert result == "No test runner found."

    @pytest.mark.asyncio
    async def test_run_all_checks_combines_sections(self, mock_client: MagicMock) -> None:
        """_run_all_checks joins sections with headers."""
        tester = TesterAgent(client=mock_client)

        # Mock find methods to return commands
        tester._find_test_command = MagicMock(return_value=["pytest"])
        tester._find_lint_command = MagicMock(return_value=["ruff"])
        tester._find_typecheck_command = MagicMock(return_value=["mypy"])

        # Mock _run_command to return different outputs for each
        async def mock_run_command(cmd, name):
            if name == "Tests":
                return "=== Tests ===\n2 passed"
            elif name == "Lint":
                return "=== Lint ===\nNo issues"
            elif name == "Typecheck":
                return "=== Typecheck ===\nSuccess"
            return None

        tester._run_command = AsyncMock(side_effect=mock_run_command)

        result = await tester._run_all_checks()

        assert "=== Tests ===" in result
        assert "2 passed" in result
        assert "=== Lint ===" in result
        assert "No issues" in result
        assert "=== Typecheck ===" in result
        assert "Success" in result

    @pytest.mark.asyncio
    async def test_run_returns_test_result(self, mock_client: MagicMock) -> None:
        """run() returns a TestResult from LLM response."""
        tester = TesterAgent(client=mock_client)

        # Mock _run_all_checks to return test output
        tester._run_all_checks = AsyncMock(return_value="=== Tests ===\n2 passed")

        result = await tester.run("Test goal", [])

        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.summary == "All good"
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_run_handles_json_decode_error(self, mock_client: MagicMock) -> None:
        """run() handles JSON decode error from LLM."""
        mock_client.complete = AsyncMock(return_value="Not valid JSON {{{")
        tester = TesterAgent(client=mock_client)
        tester._run_all_checks = AsyncMock(return_value="test output")

        result = await tester.run("Test goal", [])

        assert isinstance(result, TestResult)
        # Falls back to checking "passed" in response
        assert result.passed is False  # "passed" not in "Not valid JSON {{{"

    @pytest.mark.asyncio
    async def test_run_handles_exception_in_checks(self, mock_client: MagicMock) -> None:
        """run() handles exception in _run_all_checks."""
        tester = TesterAgent(client=mock_client)
        tester._run_all_checks = AsyncMock(side_effect=RuntimeError("Check failed"))

        result = await tester.run("Test goal", [])

        assert isinstance(result, TestResult)
        assert result.passed is False
        assert "Check failed" in result.summary
        assert result.failures == ["Check failed"]

    @pytest.mark.asyncio
    async def test_run_passes_goal_and_tasks_to_llm(self, mock_client: MagicMock) -> None:
        """run() passes goal and tasks to LLM."""
        tester = TesterAgent(client=mock_client)
        tester._run_all_checks = AsyncMock(return_value="test output")

        tasks = [
            TaskModel(id="task-1", description="Task 1"),
            TaskModel(id="task-2", description="Task 2"),
        ]

        await tester.run("Test goal", tasks)

        # Verify complete was called with goal and test output
        mock_client.complete.assert_called_once()
        call_args = mock_client.complete.call_args
        prompt = call_args[1].get("prompt", "") if call_args[1] else call_args[0][0]
        assert "Test goal" in prompt
        assert "test output" in prompt