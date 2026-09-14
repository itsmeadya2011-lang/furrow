import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient


class TestTesterAgentRun:
    @pytest.mark.asyncio
    async def test_run_returns_test_result_on_success(self):
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "passed": True,
            "summary": "all tests passed",
            "failures": [],
        }))
        mock_client.settings.tester_model = "test-tester"
        mock_client.settings.test_timeout = 120

        tester = TesterAgent(client=mock_client)

        with patch.object(TesterAgent, "_run_tests", new=AsyncMock(return_value="all tests passed")):
            result = await tester.run("test goal", [])

        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.summary == "all tests passed"
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_run_returns_fallback_on_invalid_json(self):
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(return_value="tests passed")
        mock_client.settings.tester_model = "test-tester"
        mock_client.settings.test_timeout = 120

        tester = TesterAgent(client=mock_client)

        with patch.object(TesterAgent, "_run_tests", new=AsyncMock(return_value="some output")):
            result = await tester.run("test goal", [])

        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.summary == "tests passed"

    @pytest.mark.asyncio
    async def test_run_returns_false_on_no_pass_keyword(self):
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(return_value="tests failed")
        mock_client.settings.tester_model = "test-tester"
        mock_client.settings.test_timeout = 120

        tester = TesterAgent(client=mock_client)

        with patch.object(TesterAgent, "_run_tests", new=AsyncMock(return_value="some output")):
            result = await tester.run("test goal", [])

        assert result.passed is False
        assert result.summary == "tests failed"


class TestTesterRunTests:
    @pytest.mark.asyncio
    async def test_no_test_runner_found(self):
        mock_client = MagicMock()
        mock_client.settings.test_timeout = 120

        tester = TesterAgent(client=mock_client)

        with patch("shutil.which", return_value=None):
            result = await tester._run_tests()

        assert result == "No test runner found."

    @pytest.mark.asyncio
    async def test_returns_exit_code_and_output(self):
        mock_client = MagicMock()
        mock_client.settings.test_timeout = 120

        tester = TesterAgent(client=mock_client)

        with patch("shutil.which", return_value="/usr/bin/pytest"):
            with patch("asyncio.create_subprocess_exec") as mock_exec:
                mock_proc = MagicMock()
                mock_proc.returncode = 0
                mock_exec.return_value = mock_proc

                with patch.object(mock_proc, "communicate", new=AsyncMock(return_value=(b"output", b"error"))):
                    result = await tester._run_tests()

        assert "Exit code: 0" in result
        assert "output" in result
        mock_exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_exit_code_on_failure(self):
        mock_client = MagicMock()
        mock_client.settings.test_timeout = 120

        tester = TesterAgent(client=mock_client)

        with patch("shutil.which", return_value="/usr/bin/pytest"):
            with patch("asyncio.create_subprocess_exec") as mock_exec:
                mock_proc = MagicMock()
                mock_proc.returncode = 1
                mock_exec.return_value = mock_proc

                with patch.object(mock_proc, "communicate", new=AsyncMock(return_value=(b"fail output", b""))):
                    result = await tester._run_tests()

        assert "Exit code: 1" in result
        assert "fail output" in result

    @pytest.mark.asyncio
    async def test_timeout_returns_timeout_message(self):
        mock_client = MagicMock()
        mock_client.settings.test_timeout = 5

        tester = TesterAgent(client=mock_client)

        with patch("shutil.which", return_value="/usr/bin/pytest"):
            with patch("asyncio.create_subprocess_exec") as mock_exec:
                mock_proc = MagicMock()
                mock_exec.return_value = mock_proc

                with patch.object(mock_proc, "communicate", new=AsyncMock(side_effect=asyncio.TimeoutError())):
                    result = await tester._run_tests()

        assert "timed out" in result

    @pytest.mark.asyncio
    async def test_skips_unavailable_runners(self):
        mock_client = MagicMock()
        mock_client.settings.test_timeout = 120

        tester = TesterAgent(client=mock_client)

        with patch("shutil.which", return_value=None):
            with patch("asyncio.create_subprocess_exec") as mock_exec:
                result = await tester._run_tests()

        assert result == "No test runner found."
        mock_exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_tries_multiple_runners_until_one_found(self):
        mock_client = MagicMock()
        mock_client.settings.test_timeout = 120

        tester = TesterAgent(client=mock_client)

        call_count = {"n": 0}

        def mock_which(cmd):
            call_count["n"] += 1
            if call_count["n"] < 3:
                return None
            return "/usr/bin/pytest"

        with patch("shutil.which", side_effect=mock_which):
            with patch("asyncio.create_subprocess_exec") as mock_exec:
                mock_proc = MagicMock()
                mock_proc.returncode = 0
                mock_exec.return_value = mock_proc

                with patch.object(mock_proc, "communicate", new=AsyncMock(return_value=(b"ok", b""))):
                    result = await tester._run_tests()

        assert "Exit code: 0" in result

    @pytest.mark.asyncio
    async def test_creates_subprocess_with_correct_args(self):
        mock_client = MagicMock()
        mock_client.settings.test_timeout = 120

        tester = TesterAgent(client=mock_client)

        with patch("shutil.which", return_value="/usr/bin/pytest"):
            with patch("asyncio.create_subprocess_exec") as mock_exec:
                mock_proc = MagicMock()
                mock_proc.returncode = 0
                mock_exec.return_value = mock_proc

                with patch.object(mock_proc, "communicate", new=AsyncMock(return_value=(b"ok", b""))):
                    await tester._run_tests()

        mock_exec.assert_called_once_with(
            "pytest", "-q",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    @pytest.mark.asyncio
    async def test_creates_subprocess_with_python_pytest(self):
        mock_client = MagicMock()
        mock_client.settings.test_timeout = 120

        tester = TesterAgent(client=mock_client)

        call_count = {"n": 0}

        def mock_which(cmd):
            call_count["n"] += 1
            return None if call_count["n"] == 1 else "/usr/bin/python"

        with patch("shutil.which", side_effect=mock_which):
            with patch("asyncio.create_subprocess_exec") as mock_exec:
                mock_proc = MagicMock()
                mock_proc.returncode = 0
                mock_exec.return_value = mock_proc

                with patch.object(mock_proc, "communicate", new=AsyncMock(return_value=(b"ok", b""))):
                    await tester._run_tests()

        mock_exec.assert_called_once_with(
            "python", "-m", "pytest", "-q",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    @pytest.mark.asyncio
    async def test_run_tests_handles_exception_gracefully(self):
        mock_client = MagicMock()
        mock_client.settings.test_timeout = 120

        tester = TesterAgent(client=mock_client)

        with patch("shutil.which", return_value="/usr/bin/pytest"):
            with patch("asyncio.create_subprocess_exec", side_effect=OSError("cannot run")):
                result = await tester._run_tests()

        assert result == "No test runner found."
