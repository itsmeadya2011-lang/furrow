"""Contracts for the command line interface.

Protected behaviour, one test per contract:

* ``--help`` on the group and on each subcommand exits zero on a machine with no
  credentials: the API-key check must live in the subcommands, not the group callback.
* ``--version`` reports the installed distribution version.
* Option validation: ``--max-parallel-tasks`` stays inside 1..32, ``--max-cycles`` accepts
  0 (unlimited) and rejects negatives, ``--provider`` rejects unknown providers.
* A missing API key fails fast with a message naming the environment variable and the
  ``--provider`` escape hatch, instead of dying inside an SDK constructor.
* ``--workspace`` reaches ``settings.workspace``, ``--model`` reaches every per-role model
  (the fields the agents actually read) and an explicit per-role flag wins over ``--model``.
* ``web`` forwards ``--host``/``--port`` to the server's ``run()`` without binding a socket.
* A runtime failure exits 1 with a readable message and no traceback; ``FURROW_TRACEBACK=1``
  is the documented escape hatch that does re-raise.

Every test drives the CLI through ``CliRunner`` with the orchestrator replaced by a double,
so nothing here starts a server or calls an LLM.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module, metadata
from pathlib import Path
from typing import Any, ClassVar

import pytest
from click.testing import CliRunner

from furrow.cli.main import main
from furrow.config import settings

# ``furrow.cli.main`` as a package attribute is the click Group re-exported by
# ``furrow/cli/__init__.py``, so the module itself has to be looked up by name.
cli_main = import_module("furrow.cli.main")

TRACE_MARKER = "Traceback (most recent call last)"

try:
    # The CLI prefers the installed distribution and falls back to ``furrow.__version__``,
    # so a source checkout has nothing to compare against.
    INSTALLED_VERSION: str | None = metadata.version("furrow")
except metadata.PackageNotFoundError:  # pragma: no cover - source checkout
    INSTALLED_VERSION = None


class _RecordingOrchestrator:
    """Stands in for the engine so ``start`` runs its whole body without any LLM work."""

    goals: ClassVar[list[str]] = []
    error: ClassVar[BaseException | None] = None

    def __init__(self, goal: str, **_: Any) -> None:
        _RecordingOrchestrator.goals.append(goal)
        self.goal = goal

    async def run(self) -> None:
        if _RecordingOrchestrator.error is not None:
            raise _RecordingOrchestrator.error


@pytest.fixture
def run_cli(monkeypatch: pytest.MonkeyPatch) -> Callable[..., Any]:
    """Return an invoker that reaches the orchestrator with a usable credential.

    The engine is replaced by a recording double and an Anthropic key is exported, so
    option parsing, credential checks and settings mutation all happen for real while the
    test still cannot touch the network.
    """
    _RecordingOrchestrator.goals = []
    _RecordingOrchestrator.error = None
    monkeypatch.setattr(cli_main, "Orchestrator", _RecordingOrchestrator)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    runner = CliRunner()

    def invoke(*args: str) -> Any:
        return runner.invoke(main, list(args))

    return invoke


@pytest.fixture
def no_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every credential the CLI could find, environment and settings alike."""
    for name in ("ANTHROPIC_API_KEY", "FURROW_ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(settings, "anthropic_api_key", None)


def test_help_exits_zero() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "start" in result.output


def test_start_help_exits_zero(no_credential: None) -> None:
    result = CliRunner().invoke(main, ["start", "--help"])

    assert result.exit_code == 0
    assert "--max-parallel-tasks" in result.output


def test_web_help_exits_zero(no_credential: None) -> None:
    result = CliRunner().invoke(main, ["web", "--help"])

    assert result.exit_code == 0
    assert "--host" in result.output


def test_version_reports_the_package_version() -> None:
    result = CliRunner().invoke(main, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == f"furrow, version {cli_main._resolve_version()}"
    if INSTALLED_VERSION is not None:
        assert INSTALLED_VERSION in result.output


def test_max_parallel_tasks_range_is_enforced(run_cli: Callable[..., Any]) -> None:
    result = run_cli("start", "a goal", "--max-parallel-tasks", "99")

    assert result.exit_code != 0
    assert "not in the range 1<=x<=32" in result.output


def test_max_parallel_tasks_accepts_the_range_bounds(run_cli: Callable[..., Any]) -> None:
    assert run_cli("start", "a goal", "--max-parallel-tasks", "32").exit_code == 0
    assert settings.max_parallel_tasks == 32


def test_max_cycles_accepts_zero(run_cli: Callable[..., Any]) -> None:
    result = run_cli("start", "a goal", "--max-cycles", "0")

    assert result.exit_code == 0
    assert settings.max_cycles == 0
    assert _RecordingOrchestrator.goals == ["a goal"]


def test_max_cycles_rejects_a_negative_value(run_cli: Callable[..., Any]) -> None:
    result = run_cli("start", "a goal", "--max-cycles", "-1")

    assert result.exit_code != 0
    assert "not in the range x>=0" in result.output


def test_missing_api_key_exits_nonzero_with_a_readable_message(
    no_credential: None,
) -> None:
    result = CliRunner().invoke(main, ["start", "a goal"])

    assert result.exit_code != 0
    assert "ANTHROPIC_API_KEY" in result.output
    assert "--provider" in result.output


def test_workspace_option_reaches_settings(tmp_path: Path, run_cli: Callable[..., Any]) -> None:
    result = run_cli("--workspace", str(tmp_path), "start", "a goal")

    assert result.exit_code == 0
    assert settings.workspace == tmp_path


def test_provider_choice_rejects_unknown_values(run_cli: Callable[..., Any]) -> None:
    result = run_cli("--provider", "bogus", "start", "a goal")

    assert result.exit_code != 0
    assert "bogus" in result.output


def test_model_override_propagates_to_per_role_models(run_cli: Callable[..., Any]) -> None:
    result = run_cli("start", "a goal", "--model", "override-model")

    assert result.exit_code == 0
    assert settings.model == "override-model"
    assert settings.planner_model == "override-model"
    assert settings.worker_model == "override-model"
    assert settings.tester_model == "override-model"


def test_explicit_role_model_wins_over_model_override(run_cli: Callable[..., Any]) -> None:
    result = run_cli("start", "a goal", "--model", "override-model", "--planner-model", "own")

    assert result.exit_code == 0
    assert settings.planner_model == "own"
    assert settings.worker_model == "override-model"
    assert settings.tester_model == "override-model"


def test_web_options_are_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr("furrow.web.server.run", fake_run)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    result = CliRunner().invoke(main, ["web", "--host", "0.0.0.0", "--port", "9999"])

    assert result.exit_code == 0
    assert calls == [{"host": "0.0.0.0", "port": 9999}]


def test_web_defaults_to_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("furrow.web.server.run", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    result = CliRunner().invoke(main, ["web"])

    assert result.exit_code == 0
    assert calls == [{"host": "127.0.0.1", "port": 8000}]


def test_runtime_error_exits_nonzero_without_a_traceback(run_cli: Callable[..., Any]) -> None:
    _RecordingOrchestrator.error = RuntimeError("the workspace vanished")

    result = run_cli("start", "a goal")

    assert result.exit_code == 1
    assert "the workspace vanished" in result.output
    assert TRACE_MARKER not in result.output


def test_runtime_error_reraises_with_furrow_traceback(
    run_cli: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _RecordingOrchestrator.error = RuntimeError("the workspace vanished")
    monkeypatch.setenv("FURROW_TRACEBACK", "1")

    result = run_cli("start", "a goal")

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert result.exception.args == ("the workspace vanished",)
