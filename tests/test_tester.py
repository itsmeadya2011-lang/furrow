"""Tests for the tester agent."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.tester import TEST_RUNNERS, TesterAgent
from furrow.config import Settings, TaskModel


class TestTestRunnerCandidates:
    """Tests for the test runner candidate list."""

    def test_has_pytest(self):
        runners = [name for _, name in TEST_RUNNERS]
        assert "pytest" in runners

    def test_has_multiple_runners(self):
        assert len(TEST_RUNNERS) >= 6

    def test_cargo_and_go_present(self):
        runners = [name for _, name in TEST_RUNNERS]
        assert "cargo" in runners
        assert "go" in runners


class TestTesterRunTests:
    """Tests for _run_tests."""

    @pytest.mark.asyncio
    async def test_no_test_runner_found(self, tmp_path):
        """When no test runner is available, return message."""
        settings = Settings(workspace=tmp_path)
        tester = TesterAgent(settings=settings)

        # Patch shutil.which to return None (no runners found)
        with patch("furrow.agents.tester.shutil.which", return_value=None):
            result = await tester._run_tests()

        assert result == "No test runner found."

    @pytest.mark.asyncio
    async def test_test_command_timeout(self, tmp_path):
        """Timeout should be handled gracefully."""
        settings = Settings(workspace=tmp_path)
        tester = TesterAgent(settings=settings)

        # Mock shutil.which to find a runner
        with (
            patch("furrow.agents.tester.shutil.which", return_value="/usr/bin/pytest"),
            patch("furrow.agents.tester.asyncio.create_subprocess_exec") as mock_exec,
        ):
            # Create a mock process that hangs (times out)
            mock_proc = MagicMock()
            mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_proc.kill = MagicMock()
            mock_exec.return_value = mock_proc

            result = await tester._run_tests()

        assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_test_command_error_continues(self, tmp_path):
        """Test runner errors should continue to next runner."""
        settings = Settings(workspace=tmp_path)
        tester = TesterAgent(settings=settings)

        with (
            patch("furrow.agents.tester.shutil.which", return_value="/usr/bin/pytest"),
            patch("furrow.agents.tester.asyncio.create_subprocess_exec") as mock_exec,
        ):
            # First call raises, simulate continuing
            mock_exec.side_effect = FileNotFoundError("Not found")

            result = await tester._run_tests()

        assert result == "No test runner found."


class TestTesterRun:
    """Tests for the tester's run() method."""

    @pytest.mark.asyncio
    async def test_no_test_runner_returns_passed(self, tmp_path):
        """When no test runner is found, return passed with info message."""
        settings = Settings(workspace=tmp_path)
        tester = TesterAgent(settings=settings)

        mock_client = MagicMock()
        mock_client.settings = settings
        mock_client.complete = AsyncMock(
            return_value='{"passed": true, "summary": "ok", "failures": []}'
        )
        tester.client = mock_client

        with patch("furrow.agents.tester.shutil.which", return_value=None):
            result = await tester.run("test goal", [TaskModel(id="1", description="test")])

        assert result.passed is True
        assert "No test runner" in result.summary

    @pytest.mark.asyncio
    async def test_parse_json_response(self, tmp_path):
        """Valid JSON from LLM should be parsed into TestResult."""
        settings = Settings(workspace=tmp_path)
        tester = TesterAgent(settings=settings)

        mock_client = MagicMock()
        mock_client.settings = settings
        mock_client.complete = AsyncMock(
            return_value='{"passed": true, "summary": "All tests passed", "failures": []}'
        )
        tester.client = mock_client

        test_output = ".... 5 passed in 0.1s"
        # Mock _run_tests to return test output
        with patch.object(tester, "_run_tests", new=AsyncMock(return_value=test_output)):
            result = await tester.run("test goal", [TaskModel(id="1", description="test")])

        assert result.passed is True
        assert result.summary == "All tests passed"
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_parse_json_with_failures(self, tmp_path):
        """JSON with failures should be parsed correctly."""
        settings = Settings(workspace=tmp_path)
        tester = TesterAgent(settings=settings)

        mock_client = MagicMock()
        mock_client.settings = settings
        response_json = (
            '{"passed": false, "summary": "1 failure", "failures": ["test_auth.test_login"]}'
        )
        mock_client.complete = AsyncMock(return_value=response_json)
        tester.client = mock_client

        with patch.object(tester, "_run_tests", new=AsyncMock(return_value="1 failed")):
            result = await tester.run("test goal", [TaskModel(id="1", description="test")])

        assert result.passed is False
        assert "test_auth.test_login" in result.failures

    @pytest.mark.asyncio
    async def test_fallback_on_json_error_pass(self, tmp_path):
        """Non-JSON response with 'passed' keyword should return passed."""
        settings = Settings(workspace=tmp_path)
        tester = TesterAgent(settings=settings)

        mock_client = MagicMock()
        mock_client.settings = settings
        mock_client.complete = AsyncMock(return_value="Tests passed! Everything is good.")
        tester.client = mock_client

        with patch.object(tester, "_run_tests", new=AsyncMock(return_value="test output")):
            result = await tester.run("test goal", [TaskModel(id="1", description="test")])

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_fallback_on_json_error_fail(self, tmp_path):
        """Non-JSON response with 'fail' keyword should return failed."""
        settings = Settings(workspace=tmp_path)
        tester = TesterAgent(settings=settings)

        mock_client = MagicMock()
        mock_client.settings = settings
        mock_client.complete = AsyncMock(return_value="Tests failed! Some things broke.")
        tester.client = mock_client

        with patch.object(tester, "_run_tests", new=AsyncMock(return_value="test output")):
            result = await tester.run("test goal", [TaskModel(id="1", description="test")])

        assert result.passed is False

    @pytest.mark.asyncio
    async def test_test_error_returns_failed(self, tmp_path):
        """If test runner crashes, return Failed TestResult."""
        settings = Settings(workspace=tmp_path)
        tester = TesterAgent(settings=settings)

        mock_client = MagicMock()
        mock_client.settings = settings
        mock_client.complete = AsyncMock(
            return_value='{"passed": true, "summary": "ok", "failures": []}'
        )
        tester.client = mock_client

        # Mock _run_tests to raise an exception
        with patch.object(
            tester, "_run_tests", new=AsyncMock(side_effect=RuntimeError("Runner exploded"))
        ):
            result = await tester.run("test goal", [TaskModel(id="1", description="test")])

        assert result.passed is False
        assert "Runner exploded" in result.summary
