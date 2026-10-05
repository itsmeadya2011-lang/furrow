"""Regression tests for :class:`furrow.cli.main`.

The CLI mutates the module-level ``furrow.config.settings`` singleton in place, so every test
here runs against a fake ``Orchestrator`` and relies on the autouse
``restore_settings_singleton`` fixture to undo the mutation -- otherwise one test would pin
``max_cycles`` for the whole suite.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest
from click.testing import CliRunner, Result

from furrow.config import Provider, Settings
from furrow.config import settings as singleton_settings

# ``furrow.cli.__init__`` re-exports the ``main`` click group under the name ``main``, which
# shadows the ``furrow.cli.main`` submodule attribute. Import the submodule explicitly.
cli_module = importlib.import_module("furrow.cli.main")


class FakeOrchestrator:
    """Stands in for the real orchestrator so ``start`` never calls a model."""

    created: list[FakeOrchestrator] = []

    def __init__(self, goal: str, client: object = None, on_event: object = None) -> None:
        self.goal = goal
        self.ran = False
        FakeOrchestrator.created.append(self)

    async def run(self) -> None:
        self.ran = True


@pytest.fixture(autouse=True)
def fake_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeOrchestrator.created = []
    monkeypatch.setattr(cli_module, "Orchestrator", FakeOrchestrator)


def invoke(*args: str, **kwargs: Any) -> Result:
    return CliRunner().invoke(cli_module.main, list(args), **kwargs)


# ------------------------------------------------------------------------ option wiring


def test_the_overrides_reach_the_settings_singleton() -> None:
    result = invoke(
        "start",
        "--max-cycles",
        "3",
        "--max-parallel-tasks",
        "2",
        "--provider",
        "openai",
        "--model",
        "M",
        "--planner-model",
        "PM",
        "--worker-model",
        "WM",
        "--tester-model",
        "TM",
        "do the thing",
    )

    assert result.exit_code == 0, result.output
    assert singleton_settings.max_cycles == 3
    assert singleton_settings.max_parallel_tasks == 2
    assert singleton_settings.provider is Provider.OPENAI
    assert singleton_settings.model == "M"
    assert singleton_settings.planner_model == "PM"
    assert singleton_settings.worker_model == "WM"
    assert singleton_settings.tester_model == "TM"


def test_the_goal_reaches_the_orchestrator() -> None:
    result = invoke("start", "Add JWT auth")

    assert result.exit_code == 0, result.output
    assert [item.goal for item in FakeOrchestrator.created] == ["Add JWT auth"]
    assert all(item.ran for item in FakeOrchestrator.created)


@pytest.mark.parametrize("provider", ["anthropic", "openai", "ollama"])
def test_every_documented_provider_is_accepted(provider: str) -> None:
    result = invoke("start", "--provider", provider, "GOAL")

    assert result.exit_code == 0, result.output
    assert singleton_settings.provider is Provider(provider)


def test_an_omitted_flag_leaves_the_configured_value_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(singleton_settings, "max_cycles", 7)
    monkeypatch.setattr(singleton_settings, "max_parallel_tasks", 9)

    result = invoke("start", "GOAL")

    assert result.exit_code == 0, result.output
    assert singleton_settings.max_cycles == 7
    assert singleton_settings.max_parallel_tasks == 9


def test_max_cycles_zero_is_accepted_as_unlimited() -> None:
    result = invoke("start", "--max-cycles", "0", "GOAL")

    assert result.exit_code == 0, result.output
    assert singleton_settings.max_cycles == 0


def test_a_bare_start_leaves_the_defaults_in_place() -> None:
    result = invoke("start", "GOAL")

    assert result.exit_code == 0, result.output
    assert singleton_settings.max_cycles == 0
    assert singleton_settings.provider is Provider.ANTHROPIC
    assert singleton_settings.model == Settings().model


# -------------------------------------------------------------------- rejected input


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--max-cycles", "-1", "must be 0 (unlimited) or greater"),
        ("--max-parallel-tasks", "0", "must be 1 or greater"),
        ("--max-parallel-tasks", "-3", "must be 1 or greater"),
    ],
)
def test_out_of_range_numeric_flags_are_rejected(flag: str, value: str, message: str) -> None:
    result = invoke("start", flag, value, "GOAL")

    assert result.exit_code != 0
    assert flag in result.output
    assert message in result.output


def test_a_non_numeric_max_cycles_is_rejected() -> None:
    result = invoke("start", "--max-cycles", "many", "GOAL")

    assert result.exit_code != 0
    assert "not a valid integer" in result.output


def test_an_unknown_provider_is_rejected() -> None:
    result = invoke("start", "--provider", "telepathy", "GOAL")

    assert result.exit_code != 0
    assert "not one of" in result.output


def test_rejected_input_never_starts_a_run() -> None:
    invoke("start", "--max-cycles", "-1", "GOAL")

    assert FakeOrchestrator.created == []


# --------------------------------------------------------------------------- goal input


def test_a_missing_goal_prompts_and_then_exits_cleanly() -> None:
    result = invoke("start", input="\n")

    assert result.exit_code == 1
    assert "Enter your goal for Furrow" in result.output
    assert "No goal given. Nothing to do." in result.output
    assert FakeOrchestrator.created == []


def test_a_whitespace_only_goal_argument_still_prompts() -> None:
    result = invoke("start", "   ", input="the real goal\n")

    assert result.exit_code == 0, result.output
    assert [item.goal for item in FakeOrchestrator.created] == ["the real goal"]


def test_eof_at_the_prompt_exits_cleanly() -> None:
    result = invoke("start", input="")

    assert result.exit_code == 1
    assert "No goal given. Nothing to do." in result.output


# -------------------------------------------------------------------------- help output


def test_the_group_help_lists_the_commands() -> None:
    result = invoke("--help")

    assert result.exit_code == 0
    assert "start" in result.output
    assert "web" in result.output


@pytest.mark.parametrize(
    "flag",
    [
        "--model",
        "--planner-model",
        "--worker-model",
        "--tester-model",
        "--provider",
        "--max-cycles",
        "--max-parallel-tasks",
    ],
)
def test_every_override_flag_is_documented_in_the_help(flag: str) -> None:
    result = invoke("start", "--help")

    assert result.exit_code == 0
    assert flag in result.output


def test_the_web_command_is_registered() -> None:
    result = invoke("web", "--help")

    assert result.exit_code == 0
    assert "Usage:" in result.output
