import asyncio
import tempfile
from pathlib import Path

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import Settings, TestResult


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def _make_settings(workspace: Path, command: list[str], timeout: int = 5) -> Settings:
    """Create a Settings instance with overridden values for testing."""
    return Settings(
        test_command=command,
        test_timeout_seconds=timeout,
        workspace=workspace,
        anthropic_api_key="test-key",  # Dummy key to satisfy validation
    )


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_success_exit_zero(self, temp_workspace):
        """Test TesterAgent returns passed=True when command exits with 0."""
        # Create a script that exits successfully
        script = temp_workspace / "test_success.sh"
        script.write_text("#!/bin/sh\necho 'tests passed'\nexit 0\n")
        script.chmod(0o755)

        settings = _make_settings(temp_workspace, [str(script)])
        tester = TesterAgent(settings=settings)

        result = await tester.run("goal", [])

        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.exit_code == 0
        assert "tests passed" in result.summary
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(self, temp_workspace):
        """Test TesterAgent returns passed=False when command exits with non-zero."""
        # Create a script that exits with error
        script = temp_workspace / "test_fail.sh"
        script.write_text("#!/bin/sh\necho 'test failed'\nexit 1\n")
        script.chmod(0o755)

        settings = _make_settings(temp_workspace, [str(script)])
        tester = TesterAgent(settings=settings)

        result = await tester.run("goal", [])

        assert isinstance(result, TestResult)
        assert result.passed is False
        assert result.exit_code == 1
        assert "test failed" in result.summary
        assert len(result.failures) > 0

    @pytest.mark.asyncio
    async def test_timeout(self, temp_workspace):
        """Test TesterAgent returns passed=False when command times out."""
        # Create a script that sleeps longer than the timeout
        script = temp_workspace / "test_slow.sh"
        script.write_text("#!/bin/sh\nsleep 10\n")
        script.chmod(0o755)

        settings = _make_settings(temp_workspace, [str(script)], timeout=1)
        tester = TesterAgent(settings=settings)

        result = await tester.run("goal", [])

        assert isinstance(result, TestResult)
        assert result.passed is False
        assert result.exit_code is None
        assert "timed out" in result.summary.lower()
        assert len(result.failures) > 0
        assert "timeout" in result.failures[0].lower()

    @pytest.mark.asyncio
    async def test_missing_executable(self, temp_workspace):
        """Test TesterAgent returns passed=False when executable is not found."""
        settings = _make_settings(temp_workspace, ["/nonexistent/executable/that/does/not/exist"])
        tester = TesterAgent(settings=settings)

        result = await tester.run("goal", [])

        assert isinstance(result, TestResult)
        assert result.passed is False
        assert result.exit_code is None
        assert "not found" in result.summary.lower() or "no such file" in result.summary.lower()
        assert len(result.failures) > 0

    @pytest.mark.asyncio
    async def test_empty_command(self, temp_workspace):
        """Test TesterAgent returns passed=False when test_command is empty."""
        settings = _make_settings(temp_workspace, [])
        tester = TesterAgent(settings=settings)

        result = await tester.run("goal", [])

        assert isinstance(result, TestResult)
        assert result.passed is False
        assert "not configured" in result.summary.lower() or "empty" in result.summary.lower()
        assert len(result.failures) > 0

    @pytest.mark.asyncio
    async def test_command_with_args(self, temp_workspace):
        """Test TesterAgent works with command and arguments."""
        # Create a simple Python script that we can run with python -c
        script_content = 'import sys; print("hello"); sys.exit(0)'
        settings = _make_settings(temp_workspace, ["python", "-c", script_content])
        tester = TesterAgent(settings=settings)

        result = await tester.run("goal", [])

        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.exit_code == 0
        assert "hello" in result.summary