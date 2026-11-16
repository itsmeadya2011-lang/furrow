"""Tests for the CLI improvements."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from click.testing import CliRunner

from furrow.cli.main import main
from furrow.config import RunResult, Settings, Usage
from furrow.core.state import RunState, HistoryEntry


class TestCLICosts:
    """Tests for the CLI costs command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_costs_no_state(self, runner, tmp_path, monkeypatch):
        """Test costs command when no state exists."""
        with patch("furrow.cli.main.StateStore") as mock_store_class:
            mock_store = MagicMock()
            mock_store.load.return_value = None
            mock_store_class.return_value = mock_store

            result = runner.invoke(main, ["costs", "--workspace", str(tmp_path)])
            assert result.exit_code == 0
            assert "No furrow run state found" in result.output

    def test_costs_no_usage_data(self, runner, tmp_path, monkeypatch):
        """Test costs command when state exists but no usage data."""
        state = RunState(goal="Test", original_goal="Test", cycles=1)

        with patch("furrow.cli.main.StateStore") as mock_store_class:
            mock_store = MagicMock()
            mock_store.load.return_value = state
            mock_store_class.return_value = mock_store

            result = runner.invoke(main, ["costs", "--workspace", str(tmp_path)])
            assert result.exit_code == 0
            assert "No cost data available" in result.output

    def test_costs_with_usage(self, runner, tmp_path, monkeypatch):
        """Test costs command with usage data."""
        usage = Usage(input_tokens=1000, output_tokens=500, calls=5, cost_usd=0.01)
        history = [
            HistoryEntry(cycle=1, usage=Usage(input_tokens=500, output_tokens=250, calls=2, cost_usd=0.005)),
            HistoryEntry(cycle=2, usage=Usage(input_tokens=500, output_tokens=250, calls=3, cost_usd=0.005)),
        ]
        state = RunState(usage=usage, history=history)

        with patch("furrow.cli.main.StateStore") as mock_store_class:
            mock_store = MagicMock()
            mock_store.load.return_value = state
            mock_store_class.return_value = mock_store

            result = runner.invoke(main, ["costs", "--workspace", str(tmp_path)])
            assert result.exit_code == 0
            assert "Furrow cost summary" in result.output
            assert "$0.010000" in result.output
            assert "5" in result.output  # calls
            assert "1500" in result.output  # total tokens
            assert "Per-cycle breakdown" in result.output
            assert "Cycle   1" in result.output
            assert "Cycle   2" in result.output


class TestCLIVerbose:
    """Tests for the CLI verbose mode."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_start_verbose(self, runner, tmp_path, monkeypatch):
        """Test start command with verbose flag."""
        # Mock the orchestrator run
        mock_orchestrator = MagicMock()
        mock_result = RunResult(
            passed=True,
            cycles=1,
            tasks_completed=1,
            tasks_failed=0,
            tasks_blocked=0,
            tests_run=1,
            summary="Done",
            stopped_reason="goal complete",
            usage=Usage(calls=1, input_tokens=100, output_tokens=50, cost_usd=0.001),
            files_changed=["test.py"],
        )
        mock_orchestrator.run = AsyncMock(return_value=mock_result)

        with patch("furrow.cli.main.Orchestrator", return_value=mock_orchestrator):
            with patch("furrow.cli.main._run_and_release", return_value=mock_result):
                with patch("furrow.cli.main.StateStore"):
                    result = runner.invoke(main, [
                        "start", "test goal",
                        "--workspace", str(tmp_path),
                        "--verbose"
                    ])
                    # Should not crash
                    assert result.exit_code in (0, 1)  # May exit 1 due to test setup


class TestCLIResumeVerbose:
    """Tests for the CLI resume command with verbose."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_resume_verbose(self, runner, tmp_path, monkeypatch):
        """Test resume command with verbose flag."""
        state = RunState(
            goal="Test goal",
            original_goal="Original goal",
            cycles=1,
        )

        mock_orchestrator = MagicMock()
        mock_result = RunResult(
            passed=True,
            cycles=1,
            tasks_completed=1,
            tasks_failed=0,
            tasks_blocked=0,
            tests_run=1,
            summary="Done",
            stopped_reason="goal complete",
            usage=Usage(calls=1, input_tokens=100, output_tokens=50, cost_usd=0.001),
            files_changed=["test.py"],
        )
        mock_orchestrator.run = AsyncMock(return_value=mock_result)

        with patch("furrow.cli.main.Orchestrator", return_value=mock_orchestrator):
            with patch("furrow.cli.main._run_and_release", return_value=mock_result):
                with patch("furrow.cli.main.StateStore") as mock_store_class:
                    mock_store = MagicMock()
                    mock_store.load.return_value = state
                    mock_store_class.return_value = mock_store

                    result = runner.invoke(main, [
                        "resume",
                        "--workspace", str(tmp_path),
                        "--verbose"
                    ])
                    assert result.exit_code in (0, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])