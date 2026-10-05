"""Tests for the command line entry point in :mod:`furrow.cli.main`.

Everything runs in-process through click's ``CliRunner``: no subprocesses. The
orchestrator is replaced by a stub in the only test that gets far enough to
build one; the missing-key test fails earlier than that, so no test can reach an
LLM. Provider keys are cleared for every test by the autouse fixture in
``conftest.py``, and error text is read through :func:`_cli_output` because
click 8.2 stopped mixing stderr into ``result.output``.
"""

from __future__ import annotations

from typing import Any

import pytest
from click.testing import CliRunner

import furrow.cli.main as cli_main
from furrow.cli.main import build_settings, main, missing_api_key
from furrow.config import Provider, get_settings


def _cli_output(result: object) -> str:
    """All captured output, on click 8.1 (mixed) and 8.2+ (separate streams).

    click 8.1 merges stderr into ``output`` and raises when ``stderr`` was never
    captured on its own; click 8.2 dropped ``mix_stderr`` and always captures the
    two streams apart, leaving ``output`` as stdout only. Read both so a message
    that lands on stderr is found either way.
    """
    text = getattr(result, "output", "") or ""
    try:
        stderr = getattr(result, "stderr", None)
    except Exception:  # click 8.1: stderr was not captured separately
        return text
    return text if not isinstance(stderr, str) else text + stderr


class _StubOrchestrator:
    """Records the goal it was built with and does nothing when run."""

    goals: list[str] = []

    def __init__(self, goal: str, client: Any = None, on_event: Any = None) -> None:
        _StubOrchestrator.goals.append(goal)

    async def run(self) -> None:
        return None


# --------------------------------------------------------------------------- #
# missing_api_key
# --------------------------------------------------------------------------- #
def test_ollama_never_needs_an_api_key() -> None:
    assert missing_api_key(Provider.OLLAMA, get_settings()) is None


def test_a_missing_anthropic_key_is_named() -> None:
    assert missing_api_key(Provider.ANTHROPIC, get_settings()) == "ANTHROPIC_API_KEY"


def test_an_anthropic_key_in_the_environment_is_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert missing_api_key(Provider.ANTHROPIC, get_settings()) is None


def test_a_missing_openai_key_is_named_and_found(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    assert missing_api_key(Provider.OPENAI, settings) == "OPENAI_API_KEY"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert missing_api_key(Provider.OPENAI, settings) is None


# --------------------------------------------------------------------------- #
# build_settings
# --------------------------------------------------------------------------- #
def test_build_settings_does_not_mutate_the_process_settings() -> None:
    before = get_settings()
    updated = build_settings(model="m1")
    assert updated.model == "m1"
    assert updated is not before
    # The singleton is untouched: the old code did ``settings.model = model``.
    assert get_settings() is before
    assert before.model == "claude-sonnet-4-20250514"


def test_build_settings_without_options_keeps_the_defaults() -> None:
    base = get_settings()
    unchanged = build_settings()
    assert unchanged is not base
    assert unchanged.model == base.model
    assert unchanged.max_cycles == 0
    assert base.max_cycles == 0


def test_build_settings_applies_every_override() -> None:
    built = build_settings(
        model="m1",
        provider="openai",
        max_cycles=3,
        max_parallel=2,
        test_command="pytest",
    )
    assert built.model == "m1"
    assert built.provider == Provider.OPENAI
    assert built.max_cycles == 3
    assert built.max_parallel_tasks == 2
    assert built.test_command == "pytest"


def test_build_settings_never_touches_the_singleton() -> None:
    base = get_settings()
    build_settings(
        model="m1",
        provider="openai",
        max_cycles=3,
        max_parallel=2,
        test_command="pytest",
    )
    assert get_settings() is base
    assert base.model == "claude-sonnet-4-20250514"
    assert base.provider == Provider.ANTHROPIC
    assert base.max_cycles == 0
    assert base.max_parallel_tasks == 5
    assert base.test_command is None


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def test_start_reports_a_missing_api_key_before_running() -> None:
    result = CliRunner().invoke(main, ["start", "some goal", "--provider", "anthropic"])

    # The command raises ``click.UsageError``, which click's standalone mode turns
    # into a non-zero exit with the message on stderr; the exact code is click's
    # ``UsageError.exit_code``, so only "it failed" is asserted here.
    output = _cli_output(result)
    assert result.exit_code != 0, output
    assert "ANTHROPIC_API_KEY" in output


def test_start_help_lists_the_override_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "200")  # keep the option column unwrapped
    result = CliRunner().invoke(main, ["start", "--help"])
    assert result.exit_code == 0, result.output
    for option in ("--provider", "--max-cycles", "--max-parallel", "--test-command"):
        assert option in result.output


def test_web_help_lists_host_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    result = CliRunner().invoke(main, ["web", "--help"])
    assert result.exit_code == 0, result.output
    assert "--host" in result.output
    assert "--port" in result.output


def test_start_hands_the_goal_to_the_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    _StubOrchestrator.goals = []
    monkeypatch.setattr(cli_main, "Orchestrator", _StubOrchestrator)

    result = CliRunner().invoke(main, ["start", "build a thing"])

    assert result.exit_code == 0, result.output
    assert result.exception is None, repr(result.exception)
    assert _StubOrchestrator.goals == ["build a thing"]