"""Tests for the TesterAgent class."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.tester import TesterAgent


def _make_proc(output: bytes = b"output", err: bytes = b"") -> MagicMock:
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(output, err))
    proc.kill = MagicMock()
    return proc


# ---------------------------------------------------------------------------
# _run_tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_tests_includes_all_sections() -> None:
    tester = TesterAgent(client=MagicMock())
    proc = _make_proc(b"all good")

    with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)):
        output = await tester._run_tests()

    assert "=== TESTS ===" in output
    assert "=== LINT ===" in output
    assert "=== TYPE CHECK ===" in output
    assert "all good" in output


@pytest.mark.asyncio
async def test_run_tests_skips_missing_lint_and_mypy() -> None:
    tester = TesterAgent(client=MagicMock())
    test_proc = _make_proc(b"tests passed")

    async def mock_exec(*args, **kwargs):
        if args[0] in ("ruff", "mypy"):
            raise FileNotFoundError(f"{args[0]} not found")
        return test_proc

    with patch("asyncio.create_subprocess_exec", new=mock_exec):
        output = await tester._run_tests()

    assert "=== TESTS ===" in output
    assert "=== LINT ===" not in output
    assert "=== TYPE CHECK ===" not in output


@pytest.mark.asyncio
async def test_run_tests_no_runner_found() -> None:
    tester = TesterAgent(client=MagicMock())

    async def mock_exec(*args, **kwargs):
        raise FileNotFoundError()

    with patch("asyncio.create_subprocess_exec", new=mock_exec):
        output = await tester._run_tests()

    assert "=== TESTS ===" in output
    assert "No test runner found." in output


# ---------------------------------------------------------------------------
# run (LLM parsing)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_parses_json_response() -> None:
    tester = TesterAgent(client=MagicMock())
    tester.client.complete = AsyncMock(
        return_value='{"passed": true, "summary": "all good", "failures": []}'
    )
    tester._run_tests = AsyncMock(return_value="=== TESTS ===\nok")

    result = await tester.run("test goal", [])

    assert result.passed is True
    assert result.summary == "all good"
    assert result.failures == []


@pytest.mark.asyncio
async def test_run_handles_non_json_response() -> None:
    tester = TesterAgent(client=MagicMock())
    tester.client.complete = AsyncMock(return_value="Tests passed")
    tester._run_tests = AsyncMock(return_value="=== TESTS ===\nok")

    result = await tester.run("test goal", [])

    assert result.passed is True
    assert result.summary == "Tests passed"


@pytest.mark.asyncio
async def test_run_reports_exceptions() -> None:
    tester = TesterAgent(client=MagicMock())
    tester._run_tests = AsyncMock(side_effect=RuntimeError("boom"))

    result = await tester.run("test goal", [])

    assert result.passed is False
    assert "boom" in result.summary
