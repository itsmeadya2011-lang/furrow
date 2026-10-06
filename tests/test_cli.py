from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from furrow.cli.main import main
from furrow.config import Provider, Settings


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def fake_settings() -> Settings:
    return Settings(provider=Provider.ANTHROPIC, model="default-model", max_cycles=0)


def test_start_prompts_for_goal(runner: CliRunner, fake_settings: Settings) -> None:
    orchestrator = MagicMock()
    orchestrator.run = AsyncMock(return_value=None)

    settings_patch = patch("furrow.config.settings", fake_settings)
    orch_patch = patch("furrow.cli.main.Orchestrator", return_value=orchestrator)

    settings_patch.start()
    orch_patch.start()
    try:
        result = runner.invoke(main, ["start"], input="my goal\n")
    finally:
        settings_patch.stop()
        orch_patch.stop()

    assert result.exit_code == 0
    assert "my goal" in result.output
    orchestrator.run.assert_awaited_once()


def test_start_rejects_empty_goal(runner: CliRunner, fake_settings: Settings) -> None:
    with patch("furrow.config.settings", fake_settings):
        result = runner.invoke(main, ["start", ""])

    assert result.exit_code == 1
    assert "empty" in result.output.lower()


def test_start_sets_model(runner: CliRunner, fake_settings: Settings) -> None:
    orchestrator = MagicMock()
    orchestrator.run = AsyncMock(return_value=None)

    with patch("furrow.config.settings", fake_settings):
        with patch("furrow.cli.main.Orchestrator", return_value=orchestrator):
            result = runner.invoke(main, ["start", "build", "--model", "gpt-4o"])

    assert result.exit_code == 0
    assert fake_settings.model == "gpt-4o"


def test_start_sets_cycles(runner: CliRunner, fake_settings: Settings) -> None:
    orchestrator = MagicMock()
    orchestrator.run = AsyncMock(return_value=None)

    with patch("furrow.config.settings", fake_settings):
        with patch("furrow.cli.main.Orchestrator", return_value=orchestrator):
            result = runner.invoke(main, ["start", "build", "--cycles", "5"])

    assert result.exit_code == 0
    assert fake_settings.max_cycles == 5


def test_start_sets_provider(runner: CliRunner, fake_settings: Settings) -> None:
    orchestrator = MagicMock()
    orchestrator.run = AsyncMock(return_value=None)

    with patch("furrow.config.settings", fake_settings):
        with patch("furrow.cli.main.Orchestrator", return_value=orchestrator):
            result = runner.invoke(main, ["start", "build", "--provider", "openai"])

    assert result.exit_code == 0
    assert fake_settings.provider == Provider.OPENAI


def test_version(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_web_command(runner: CliRunner) -> None:
    with patch("furrow.web.server.run") as run_mock:
        result = runner.invoke(main, ["web"])

    assert result.exit_code == 0
    run_mock.assert_called_once_with()