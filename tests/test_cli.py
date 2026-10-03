"""The ``furrow`` command line: flag plumbing, help text, and exit codes.

Two hazards are neutralised here:

1. Every ``start`` test drives the command against a stubbed orchestrator, so no
   test can reach a provider, run a subprocess, or depend on the working
   directory; ``no_network`` additionally makes an accidental SDK call an error
   rather than a slow hang.
2. ``start`` mutates the module-level ``settings`` singleton in place, so an
   autouse fixture snapshots every field the command can touch and puts it back
   after the test. Without it a ``--provider openai`` test would silently
   reconfigure every later test file.
"""

from __future__ import annotations

import asyncio
import importlib
import re
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from furrow import llm as llm_module
from furrow.cli.main import main
from furrow.config import Provider, Settings
from furrow.core import orchestrator as orchestrator_module

# ``furrow.cli.__init__`` re-exports the ``main`` *group*, which shadows the
# submodule attribute, so the module object has to be imported explicitly: the
# flags are patched on ``main``'s module namespace, not on the group.
cli_module = importlib.import_module("furrow.cli.main")

# Every field ``--model`` has to reach: each agent is handed its own model, so
# setting only ``Settings.model`` leaves the planner, workers and tester on their
# configured defaults and the flag is a no-op.
MODEL_FIELDS = ("model", "planner_model", "worker_model", "tester_model")

# Everything ``start`` writes to the shared singleton.
MUTATED_FIELDS = (
    "provider",
    "model",
    "planner_model",
    "worker_model",
    "tester_model",
    "max_cycles",
    "max_parallel_tasks",
    "max_idle_cycles",
    "workspace",
    "test_command",
)

# A value nothing else in the suite uses, so a leaked override is unmistakable.
MODEL = "my-model"


# ------------------------------------------------------------------ utilities


def _snapshot_settings() -> list[tuple[Any, dict[str, Any]]]:
    """Capture the current value of every field ``start`` can write."""
    targets: list[Any] = [cli_module.settings]
    if config_settings() is not cli_module.settings:
        targets.append(config_settings())
    return [
        (target, {name: getattr(target, name) for name in MUTATED_FIELDS}) for target in targets
    ]


def _restore_settings(snapshots: list[tuple[Any, dict[str, Any]]]) -> None:
    """Undo a :func:`_snapshot_settings`, on whichever singleton is live now."""
    for target, values in snapshots:
        for name, value in values.items():
            setattr(target, name, value)


def config_settings() -> Any:
    """The current ``furrow.config.settings``, read through the module each time.

    ``test_config.py`` may rebind that attribute, so a ``from ... import settings``
    at module scope could hold a stale object.
    """
    from furrow import config

    return config.settings


def install_loop(
    monkeypatch: pytest.MonkeyPatch,
    *,
    run: Callable[[Any], Awaitable[None]] | None = None,
) -> list[Any]:
    """Replace the orchestrator with a stub; returns the stubs that get built.

    The stub snapshots the settings singleton *as the command left it*, which is
    the only place a flag's effect on the loop is observable. ``run`` is awaited
    by ``Orchestrator.run`` when supplied, so a test can make the loop fail, hang
    or emit without any real work.
    """
    made: list[Any] = []

    class StubLoop:
        def __init__(self, goal: str, client: Any = None, on_event: Any = None) -> None:
            self.goal = goal
            self.client = client
            self.on_event = on_event
            self.settings = cli_module.settings.model_copy(deep=True)
            made.append(self)

        async def run(self) -> None:
            if run is not None:
                await run(self)

    # The command imported the symbol, so patching the source module alone would
    # not be seen; patch both to keep the test honest either way.
    monkeypatch.setattr(cli_module, "Orchestrator", StubLoop)
    monkeypatch.setattr(orchestrator_module, "Orchestrator", StubLoop)
    return made


async def _no_completion(self: Any, *args: Any, **kwargs: Any) -> str:
    raise AssertionError("a CLI test tried to call a provider")


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any provider access an immediate failure instead of a real request."""

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a CLI test tried to construct a provider SDK client")

    monkeypatch.setattr(llm_module.LLMClient, "complete", _no_completion)
    monkeypatch.setattr(llm_module, "AsyncAnthropic", boom)
    monkeypatch.setattr(llm_module, "AsyncOpenAI", boom)


@pytest.fixture(autouse=True)
def restore_settings() -> Iterator[None]:
    """Keep the settings singleton from leaking between tests."""
    snapshots = _snapshot_settings()
    yield
    _restore_settings(snapshots)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------- help


def test_group_help_lists_the_commands(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "start" in result.output
    assert "web" in result.output


def test_start_help_advertises_every_flag(runner: CliRunner) -> None:
    # Regression guard: a flag can exist in the command body but not be reachable.
    result = runner.invoke(main, ["start", "--help"])
    assert result.exit_code == 0
    for flag in (
        "--model",
        "--provider",
        "--max-cycles",
        "--max-parallel-tasks",
        "--max-idle-cycles",
        "--workspace",
        "--test-command",
    ):
        assert flag in result.output


def test_web_help_advertises_host_and_port(runner: CliRunner) -> None:
    result = runner.invoke(main, ["web", "--help"])
    assert result.exit_code == 0
    assert "--host" in result.output
    assert "--port" in result.output


def test_version_prints_a_version(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    # Either the installed distribution version or the source-checkout fallback.
    assert re.search(r"\d+\.\d+\S*", result.output)


# --------------------------------------------------------------------- start


def test_start_prompts_for_a_missing_goal(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    stubs = install_loop(monkeypatch)
    result = runner.invoke(main, ["start"], input="typed goal\n")
    assert result.exit_code == 0
    assert "Enter your goal" in result.output
    assert [stub.goal for stub in stubs] == ["typed goal"]


def test_start_model_flag_reaches_every_agent(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    # THE regression: ``--model`` used to set only ``Settings.model``, which no
    # agent reads (they pass their own per-agent model), so the flag did nothing.
    stubs = install_loop(monkeypatch)
    result = runner.invoke(main, ["start", "goal", "--model", MODEL])
    assert result.exit_code == 0, result.output
    assert len(stubs) == 1
    for field in MODEL_FIELDS:
        assert getattr(stubs[0].settings, field) == MODEL, f"--model never reached {field}"


def test_start_without_a_model_flag_leaves_models_alone(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = {field: getattr(config_settings(), field) for field in MODEL_FIELDS}
    stubs = install_loop(monkeypatch)
    result = runner.invoke(main, ["start", "goal"])
    assert result.exit_code == 0, result.output
    for field in MODEL_FIELDS:
        assert getattr(stubs[0].settings, field) == before[field]


def test_start_provider_flag_lands_on_settings(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    stubs = install_loop(monkeypatch)
    result = runner.invoke(main, ["start", "goal", "--provider", "openai"])
    assert result.exit_code == 0, result.output
    assert stubs[0].settings.provider is Provider.OPENAI


def test_start_rejects_an_unknown_provider(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    stubs = install_loop(monkeypatch)
    result = runner.invoke(main, ["start", "goal", "--provider", "nope"])
    assert result.exit_code != 0
    assert stubs == []


def test_start_max_cycles_flag_lands_on_settings(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    stubs = install_loop(monkeypatch)
    result = runner.invoke(main, ["start", "goal", "--max-cycles", "7"])
    assert result.exit_code == 0, result.output
    assert stubs[0].settings.max_cycles == 7


def test_start_max_parallel_tasks_flag_lands_on_settings(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    stubs = install_loop(monkeypatch)
    result = runner.invoke(main, ["start", "goal", "--max-parallel-tasks", "3"])
    assert result.exit_code == 0, result.output
    assert stubs[0].settings.max_parallel_tasks == 3


def test_start_max_idle_cycles_flag_lands_on_settings(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    stubs = install_loop(monkeypatch)
    result = runner.invoke(main, ["start", "goal", "--max-idle-cycles", "2"])
    assert result.exit_code == 0, result.output
    assert stubs[0].settings.max_idle_cycles == 2


def test_start_workspace_flag_lands_on_settings(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stubs = install_loop(monkeypatch)
    result = runner.invoke(main, ["start", "goal", "--workspace", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert stubs[0].settings.workspace == tmp_path


def test_start_test_command_flag_lands_on_settings(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    stubs = install_loop(monkeypatch)
    result = runner.invoke(main, ["start", "goal", "--test-command", "pytest -q"])
    assert result.exit_code == 0, result.output
    assert stubs[0].settings.test_command == "pytest -q"


async def _interrupt(loop: Any) -> None:
    raise KeyboardInterrupt


async def _cancelled(loop: Any) -> None:
    raise asyncio.CancelledError


async def _boom(loop: Any) -> None:
    raise RuntimeError("boom")


def test_start_absorbs_a_keyboard_interrupt(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    stubs = install_loop(monkeypatch, run=_interrupt)
    result = runner.invoke(main, ["start", "goal"])
    assert result.exit_code == 0
    assert "stopped by user" in result.output
    assert len(stubs) == 1


def test_start_absorbs_a_cancelled_run(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    stubs = install_loop(monkeypatch, run=_cancelled)
    result = runner.invoke(main, ["start", "goal"])
    assert result.exit_code == 0
    assert "stopped by user" in result.output
    assert len(stubs) == 1


def test_start_reports_a_failure_and_exits_nonzero(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_loop(monkeypatch, run=_boom)
    result = runner.invoke(main, ["start", "goal"])
    assert result.exit_code == 1
    assert "Furrow failed" in result.output
    assert "boom" in result.output


def test_settings_snapshot_and_restore_round_trips(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The autouse restore fixture itself: it must undo what ``start`` wrote."""
    live = config_settings()
    assert live.provider is not Provider.OPENAI, "an earlier test leaked provider=openai"
    original = live.provider
    snapshots = _snapshot_settings()

    install_loop(monkeypatch)
    result = runner.invoke(main, ["start", "goal", "--provider", "openai", "--model", MODEL])
    assert result.exit_code == 0, result.output
    assert config_settings().provider is Provider.OPENAI

    _restore_settings(snapshots)
    assert config_settings().provider is original
    assert isinstance(config_settings(), Settings)


# ----------------------------------------------------------------------- web


def test_web_passes_host_and_port_to_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    from furrow.web import server as server_module

    calls: list[dict[str, Any]] = []

    def fake_run(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(server_module, "run", fake_run)
    result = CliRunner().invoke(main, ["web", "--host", "127.0.0.1", "--port", "9123"])
    assert result.exit_code == 0, result.output
    assert calls == [{"host": "127.0.0.1", "port": 9123}]
