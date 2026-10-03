"""Tests for the click CLI.

Two things are worth protecting here:

1. The module must stay importable (and cheap to import) -- ``Orchestrator`` is
   imported *inside* the ``start`` command body so that ``furrow --help`` and
   ``furrow web`` do not drag the provider SDKs in. Importing this test module
   at all is a smoke test for that refactor.
2. ``--model`` must not mutate the process-wide ``settings`` singleton. The CLI
   copies it with ``model_copy(update=...)`` and hands the copy to the
   orchestrator; mutating the singleton in place would leak one run's model
   override into every later run in the same process.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

import furrow.cli.main as cli_main
import furrow.config

# Patching note: `furrow.cli.main` does NOT bind an `Orchestrator` name at module
# scope -- `start()` does `from furrow.core.orchestrator import Orchestrator`
# inside its own body (keeping the import lazy). So `monkeypatch.setattr` on
# "furrow.cli.main.Orchestrator" would raise AttributeError, and the correct
# target is the attribute on the module the function actually imports from:
# "furrow.core.orchestrator.Orchestrator".
ORCHESTRATOR_TARGET = "furrow.core.orchestrator.Orchestrator"


class RecordingOrchestrator:
    """Fake Orchestrator that records how the CLI constructed it."""

    calls: list[dict] = []

    def __init__(self, goal: str, client=None, settings=None, **kwargs) -> None:
        self.goal = goal
        self.client = client
        self.settings = settings
        self.runs = 0
        RecordingOrchestrator.calls.append(
            {"goal": goal, "settings": settings, "client": client, "kwargs": kwargs}
        )

    async def run(self) -> None:
        self.runs += 1


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Patch the orchestrator class the CLI actually resolves, and record calls."""
    RecordingOrchestrator.calls = []
    monkeypatch.setattr(ORCHESTRATOR_TARGET, RecordingOrchestrator)
    return RecordingOrchestrator.calls


# --------------------------------------------------------------------------- #
# 1. The group is invokable and its help renders.
# --------------------------------------------------------------------------- #


def test_cli_group_help_succeeds() -> None:
    result = CliRunner().invoke(cli_main.main, ["--help"])

    assert result.exit_code == 0, f"exit={result.exit_code} exc={result.exception!r}"
    assert "start" in result.output
    assert "web" in result.output


def test_start_command_help_succeeds() -> None:
    result = CliRunner().invoke(cli_main.main, ["start", "--help"])

    assert result.exit_code == 0, f"exit={result.exit_code} exc={result.exception!r}"
    assert "--model" in result.output


# --------------------------------------------------------------------------- #
# 2. `furrow start "<goal>"` reaches the orchestrator with that goal.
# --------------------------------------------------------------------------- #


def test_start_passes_the_goal_through_to_the_orchestrator(recorded: list[dict]) -> None:
    result = CliRunner().invoke(cli_main.main, ["start", "add a healthcheck endpoint"])

    assert result.exit_code == 0, f"exit={result.exit_code} exc={result.exception!r}"
    assert result.exception is None
    assert len(recorded) == 1, "the orchestrator must be constructed exactly once"
    assert recorded[0]["goal"] == "add a healthcheck endpoint"


# --------------------------------------------------------------------------- #
# 3. --model overrides the run without touching the settings singleton.
# --------------------------------------------------------------------------- #


def test_model_option_does_not_mutate_the_settings_singleton(recorded: list[dict]) -> None:
    singleton = furrow.config.settings
    model_before = singleton.model

    result = CliRunner().invoke(cli_main.main, ["start", "do a thing", "--model", "some-model"])

    assert result.exit_code == 0, f"exit={result.exit_code} exc={result.exception!r}"
    assert len(recorded) == 1
    handed = recorded[0]["settings"]
    assert handed is not None, "the orchestrator must be given a Settings object"
    assert handed.model == "some-model", "the run must use the requested model"
    assert handed is not singleton, "the CLI must pass a copy, not the singleton"
    assert furrow.config.settings is singleton
    assert furrow.config.settings.model == model_before, (
        "--model must not leak into the process-wide settings singleton"
    )
