"""Tests for furrow.cli.main — CLI commands and model override."""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from furrow.cli.main import main
from furrow.config import Settings


def test_cli_main_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "start" in result.output
    assert "web" in result.output


def test_cli_start_help():
    runner = CliRunner()
    result = runner.invoke(main, ["start", "--help"])
    assert result.exit_code == 0
    assert "goal" in result.output.lower()
    assert "--model" in result.output
    assert "--max-cycles" in result.output


def test_cli_start_prompts_when_goal_missing(monkeypatch: pytest.MonkeyPatch):
    """Without a goal argument, the CLI should prompt the user.

    We patch Orchestrator.run and asyncio.run to avoid launching the loop.
    """
    runner = CliRunner()

    async def _noop_coro():
        return []

    monkeypatch.setattr("furrow.cli.main.Orchestrator.run", lambda self: _noop_coro())
    # asyncio.run receives a coroutine; close it immediately so it doesn't
    # trigger an "unawaited coroutine" warning.
    def _run_stub(coro):
        coro.close()
    monkeypatch.setattr("furrow.cli.main.asyncio.run", _run_stub)
    result = runner.invoke(main, ["start"], input="My CLI goal\n")
    assert result.exit_code == 0
    assert "My CLI goal" in result.output


def test_apply_model_override_sets_all_models(monkeypatch: pytest.MonkeyPatch):
    """The --model flag should update model, planner_model, worker_model, tester_model."""
    from furrow.cli.main import _apply_model_override
    settings = Settings(provider="openai")
    monkeypatch.setattr("furrow.config.settings", settings)
    _apply_model_override("gpt-4o")
    assert settings.model == "gpt-4o"
    assert settings.planner_model == "gpt-4o"
    assert settings.worker_model == "gpt-4o"
    assert settings.tester_model == "gpt-4o"


def test_apply_model_override_none_is_noop(monkeypatch: pytest.MonkeyPatch):
    """When model is None, nothing should change."""
    from furrow.cli.main import _apply_model_override
    settings = Settings()
    original = settings.model
    _apply_model_override(None)
    assert settings.model == original


def test_cli_web_help():
    runner = CliRunner()
    result = runner.invoke(main, ["web", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--host" in result.output
    assert "--port" in result.output
