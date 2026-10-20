"""Tests for the furrow command line interface (furrow.cli.main)."""

from __future__ import annotations

import importlib
import threading
from pathlib import Path
from typing import ClassVar

import pytest
from click.testing import CliRunner, Result

from furrow.config import Provider, settings

cli_main = importlib.import_module("furrow.cli.main")

runner = CliRunner()

MUTATED_FIELDS = (
    "model",
    "provider",
    "max_cycles",
    "workspace",
    "planner_model",
    "worker_model",
    "tester_model",
)


class FakeOrchestrator:
    """Stand-in for furrow.core.orchestrator.Orchestrator (never calls an LLM)."""

    goals: ClassVar[list[str]] = []

    def __init__(self, goal: str) -> None:
        self.goal = goal
        FakeOrchestrator.goals.append(goal)

    async def run(self) -> None:
        return None


@pytest.fixture
def fake_orchestrator(monkeypatch):
    FakeOrchestrator.goals = []
    monkeypatch.setattr(cli_main, "Orchestrator", FakeOrchestrator)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    return FakeOrchestrator


@pytest.fixture
def pristine_settings(monkeypatch):
    """Snapshot the settings fields the CLI mutates so they are restored afterwards."""
    for field in MUTATED_FIELDS:
        monkeypatch.setattr(settings, field, getattr(settings, field))
    return settings


def test_main_help_lists_commands() -> None:
    result = runner.invoke(cli_main.main, ["--help"])
    assert result.exit_code == 0
    assert "start" in result.output
    assert "web" in result.output


def test_start_help_lists_options() -> None:
    result = runner.invoke(cli_main.main, ["start", "--help"])
    assert result.exit_code == 0
    for option in ("--model", "--provider", "--max-cycles", "--workspace"):
        assert option in result.output


def test_web_help_lists_options() -> None:
    result = runner.invoke(cli_main.main, ["web", "--help"])
    assert result.exit_code == 0
    assert "--host" in result.output
    assert "--port" in result.output


def test_start_applies_model_override(fake_orchestrator, pristine_settings) -> None:
    result = runner.invoke(cli_main.main, ["start", "do the thing", "--model", "my-model"])
    assert result.exit_code == 0, result.output
    assert settings.model == "my-model"
    assert FakeOrchestrator.goals == ["do the thing"]


def test_start_applies_provider_override(fake_orchestrator, pristine_settings) -> None:
    result = runner.invoke(cli_main.main, ["start", "do the thing", "--provider", "ollama"])
    assert result.exit_code == 0, result.output
    assert settings.provider == Provider.OLLAMA


def test_start_applies_max_cycles_override(fake_orchestrator, pristine_settings) -> None:
    result = runner.invoke(cli_main.main, ["start", "do the thing", "--max-cycles", "3"])
    assert result.exit_code == 0, result.output
    assert settings.max_cycles == 3


def test_start_applies_workspace_override(fake_orchestrator, pristine_settings) -> None:
    result = runner.invoke(
        cli_main.main, ["start", "do the thing", "--workspace", "/tmp/furrow-ws"]
    )
    assert result.exit_code == 0, result.output
    assert settings.workspace == Path("/tmp/furrow-ws")


def test_start_without_goal_does_not_hang(fake_orchestrator) -> None:
    outcome: dict[str, Result] = {}

    def invoke() -> None:
        outcome["result"] = runner.invoke(cli_main.main, ["start"], input="")

    thread = threading.Thread(target=invoke, daemon=True)
    thread.start()
    thread.join(timeout=15)
    assert not thread.is_alive(), "CLI hung while waiting for a goal"

    result = outcome["result"]
    assert isinstance(result.exception, SystemExit)
    assert "Traceback" not in result.output
    assert FakeOrchestrator.goals == []


def test_web_command_forwards_host_and_port(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(host: str, port: int) -> None:
        calls.append({"host": host, "port": port})

    monkeypatch.setattr("furrow.web.server.run", fake_run)
    result = runner.invoke(cli_main.main, ["web", "--host", "127.0.0.1", "--port", "9123"])
    assert result.exit_code == 0, result.output
    assert calls == [{"host": "127.0.0.1", "port": 9123}]


def test_missing_credential_helper(monkeypatch, pristine_settings) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    assert cli_main._missing_credential(Provider.ANTHROPIC) == "ANTHROPIC_API_KEY"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert cli_main._missing_credential(Provider.ANTHROPIC) is None

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", None)
    assert cli_main._missing_credential(Provider.OPENAI) == "OPENAI_API_KEY"
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert cli_main._missing_credential(Provider.OPENAI) is None

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert cli_main._missing_credential(Provider.OLLAMA) is None


def test_start_missing_credential_exits_1(monkeypatch, pristine_settings) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    monkeypatch.setattr(settings, "openai_api_key", None)
    result = runner.invoke(cli_main.main, ["start", "goal"])
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.output
    assert "Traceback (most recent call last)" not in result.output


def test_web_command_uses_default_host_and_port(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(host: str, port: int) -> None:
        calls.append({"host": host, "port": port})

    monkeypatch.setattr("furrow.web.server.run", fake_run)
    result = runner.invoke(cli_main.main, ["web"])
    assert result.exit_code == 0, result.output
    assert calls == [{"host": "0.0.0.0", "port": 8000}]
