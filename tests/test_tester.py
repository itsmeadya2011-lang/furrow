"""Tests for the TesterAgent."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import TestResult


def _make_client_and_settings():
    settings = MagicMock(
        max_parallel_tasks=5,
        max_cycles=0,
        max_tokens=4096,
        provider="anthropic",
        tester_model="claude-3-5-sonnet",
    )
    client = MagicMock()
    client.settings = settings
    return client, settings


# ---------------------------------------------------------------------------
# _get_test_commands
# ---------------------------------------------------------------------------

def test_get_test_commands_custom_command_prepended(monkeypatch):
    """FURROW_TEST_COMMAND should be inserted at the front of the list."""
    monkeypatch.setenv("FURROW_TEST_COMMAND", "make test")
    client, _ = _make_client_and_settings()
    tester = TesterAgent(client=client)
    commands = tester._get_test_commands()
    assert commands[0] == ["make", "test"]


def test_get_test_commands_default():
    client, _ = _make_client_and_settings()
    tester = TesterAgent(client=client)
    commands = tester._get_test_commands()
    assert commands[0] == ["pytest", "-q"]
    assert ["cargo", "test", "-q"] in commands


# ---------------------------------------------------------------------------
# _run_tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_tests_returns_output_on_nonzero_exit():
    """Even when a test runner exits non-zero, its output should be captured."""
    client, _ = _make_client_and_settings()
    tester = TesterAgent(client=client)

    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(
        return_value=(b"FAILURES\nFAILED test_foo\n", b"")
    )
    mock_proc.returncode = 1

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        output = await tester._run_tests()

    assert "FAILURES" in output
    assert "FAILED test_foo" in output


@pytest.mark.asyncio
async def test_run_tests_timeout_kills_and_reaps():
    """On timeout the process must be killed and awaited to avoid zombies."""
    client, _ = _make_client_and_settings()
    tester = TesterAgent(client=client)
    tester._test_timeout = 0.01

    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
    mock_proc.kill = MagicMock()
    mock_proc.wait = AsyncMock(return_value=0)
    mock_proc.stdout = b"partial"
    mock_proc.stderr = b""

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        output = await tester._run_tests()

    mock_proc.kill.assert_called_once()
    mock_proc.wait.assert_awaited_once()
    assert "timed out" in output


@pytest.mark.asyncio
async def test_run_tests_file_not_found_falls_through():
    """If no test command is found, return 'No test runner found.'"""
    client, _ = _make_client_and_settings()
    tester = TesterAgent(client=client)

    with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
        output = await tester._run_tests()

    assert output == "No test runner found."


# ---------------------------------------------------------------------------
# run (full flow)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_parses_json_test_result():
    """When the LLM returns JSON, it should be parsed into a TestResult."""
    client, _ = _make_client_and_settings()
    tester = TesterAgent(client=client)

    mock_result = {
        "passed": True,
        "summary": "All tests passed",
        "failures": [],
    }

    tester._run_tests = AsyncMock(return_value="test output")
    client.complete = AsyncMock(return_value=json.dumps(mock_result))

    result = await tester.run("add auth", [])
    assert isinstance(result, TestResult)
    assert result.passed is True
    assert result.summary == "All tests passed"
    assert result.failures == []


@pytest.mark.asyncio
async def test_run_non_json_falls_back_to_text():
    """When the LLM doesn't return JSON, fall back to text parsing."""
    client, _ = _make_client_and_settings()
    tester = TesterAgent(client=client)

    tester._run_tests = AsyncMock(return_value="test output")
    client.complete = AsyncMock(return_value="All tests PASSED successfully")

    result = await tester.run("add auth", [])
    assert result.passed is True
    assert result.failures == []


@pytest.mark.asyncio
async def test_run_failure_from_test_runner():
    """If the test runner itself errors, return a failed TestResult."""
    client, _ = _make_client_and_settings()
    tester = TesterAgent(client=client)

    tester._run_tests = AsyncMock(side_effect=RuntimeError("pytest crashed"))

    result = await tester.run("add auth", [])
    assert result.passed is False
    assert "pytest crashed" in result.summary
    assert "pytest crashed" in result.failures
