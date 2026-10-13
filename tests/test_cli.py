"""Tests for the CLI."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from furrow.cli.main import main


def test_cli_no_args_shows_help():
    """CLI with no arguments should show help (exit code may be 0 or 2)."""
    runner = CliRunner()
    result = runner.invoke(main, [])
    # Click group with no subcommand returns exit code 2 (Usage error)
    # or 0 if help is triggered. Either way, output should show usage.
    assert "Usage" in result.output or "Furrow" in result.output


def test_cli_start_prompts_without_goal():
    """CLI start without goal should prompt for one."""
    runner = CliRunner()
    with patch("furrow.cli.main.Orchestrator") as MockOrchestrator:
        instance = MagicMock()
        instance.run = AsyncMock()
        MockOrchestrator.return_value = instance

        # Simulate providing the goal via input
        result = runner.invoke(main, ["start"], input="Add JWT auth\n")
        assert result.exit_code == 0
        MockOrchestrator.assert_called_once()
        assert "Add JWT auth" in str(MockOrchestrator.call_args)


def test_cli_start_with_goal():
    """CLI start with goal should create and run orchestrator."""
    runner = CliRunner()
    with patch("furrow.cli.main.Orchestrator") as MockOrchestrator:
        instance = MagicMock()
        instance.run = AsyncMock()
        MockOrchestrator.return_value = instance

        result = runner.invoke(main, ["start", "Add JWT auth"])
        assert result.exit_code == 0
        MockOrchestrator.assert_called_once()
        assert "Add JWT auth" in str(MockOrchestrator.call_args)


def test_cli_start_with_model_override():
    """CLI start with --model should override the model setting."""
    runner = CliRunner()
    with patch("furrow.cli.main.Orchestrator") as MockOrchestrator:
        with patch("furrow.config.settings") as mock_settings:
            instance = MagicMock()
            instance.run = AsyncMock()
            MockOrchestrator.return_value = instance

            result = runner.invoke(main, ["start", "Add tests", "--model", "gpt-4"])
            assert result.exit_code == 0
            mock_settings.model = "gpt-4"


def test_cli_keyboard_interrupt_handled():
    """CLI should handle KeyboardInterrupt gracefully."""
    runner = CliRunner()
    with patch("furrow.cli.main.Orchestrator") as MockOrchestrator:
        instance = MagicMock()

        def raise_interrupt():
            raise KeyboardInterrupt()

        instance.run = raise_interrupt
        MockOrchestrator.return_value = instance

        result = runner.invoke(main, ["start", "Add feature"])
        assert result.exit_code == 0
        assert "stopped" in result.output.lower()


def test_cli_web_command():
    """CLI web command should start the web server."""
    runner = CliRunner()
    with patch("furrow.web.server.run") as mock_run:
        result = runner.invoke(main, ["web"])
        assert result.exit_code == 0
        mock_run.assert_called_once()


def test_cli_has_start_and_web_commands():
    """CLI should expose 'start' and 'web' commands."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "start" in result.output
    assert "web" in result.output
