"""Contracts for :mod:`furrow.cli.main`.

The CLI has exactly one job: fold flags into the global
:data:`furrow.config.settings` object and hand off. Everything asserted here is
about that translation -- which model ids reach which role, which provider they
belong to, which keywords the ``web`` command forwards -- plus the two ways a
command is allowed to fail loudly and cleanly (missing credentials, and an
expected ``ValueError``) instead of dumping a traceback and exiting 0.

Both collaborators are patched where the command imports them, because ``start``
and ``web`` import lazily: nothing here starts a server or a real agent run.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from furrow import config
from furrow.cli.main import MODEL_FIELDS, PROVIDER_DEFAULT_MODELS, main
from furrow.config import Provider

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _everything(result: Any) -> str:
    """stdout + stderr.

    Click 8.5 keeps them apart (``Result.output`` is stdout only), so error
    messages that Click writes to stderr need ``Result.stderr`` as well.
    """
    text = result.output or ""
    # ``Result.stderr`` raises when Click did not capture stderr separately.
    with contextlib.suppress(ValueError):
        text += result.stderr or ""
    return text


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def global_settings() -> Any:
    """The CLI mutates the global singleton in place; ``restore_global_settings`` undoes it.

    Credentials are set because the sandbox has none: ``start`` refuses to begin a
    run without a key for the effective provider.
    """
    settings = config.settings
    settings.anthropic_api_key = "test-key"
    settings.openai_api_key = "test-key"
    return settings


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the lazily imported ``Orchestrator`` with a recorder."""
    captured: dict[str, Any] = {}

    class _RecordingOrchestrator:
        def __init__(self, goal: str, settings: Any = None) -> None:
            captured["goal"] = goal
            captured["settings"] = settings

        async def run(self) -> None:
            captured["ran"] = True

    monkeypatch.setattr("furrow.core.orchestrator.Orchestrator", _RecordingOrchestrator)
    return captured


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the lazily imported ``furrow.web.server.run`` with a kwargs recorder."""
    server = pytest.importorskip("furrow.web.server")
    captured: dict[str, Any] = {}

    def _run(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(server, "run", _run)
    return captured


# --------------------------------------------------------------------------
# --version / --help
# --------------------------------------------------------------------------


def test_version_prints_the_distribution_version(runner: CliRunner) -> None:
    """``furrow, version 0.0.0`` from a source checkout; the real id once installed."""
    result = runner.invoke(main, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip().startswith("furrow, version ")
    version = result.output.strip().removeprefix("furrow, version ")
    assert version, "no version in --version output"
    assert version == "0.0.0" or version.count(".") >= 1


def test_help_lists_both_commands(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "start" in result.output
    assert "web" in result.output


def test_start_help_lists_the_runtime_options(runner: CliRunner) -> None:
    result = runner.invoke(main, ["start", "--help"])

    assert result.exit_code == 0
    for flag in ("--model", "--provider", "--workspace", "--log-level", "--max-cycles"):
        assert flag in result.output


def test_web_help_lists_the_server_options(runner: CliRunner) -> None:
    result = runner.invoke(main, ["web", "--help"])

    assert result.exit_code == 0
    for flag in ("--host", "--port", "--reload"):
        assert flag in result.output


# --------------------------------------------------------------------------
# No goal -> actionable error, no traceback
# --------------------------------------------------------------------------


def test_an_empty_goal_is_an_actionable_error_not_a_traceback(runner: CliRunner) -> None:
    """EOF on an empty stdin must say how to supply a goal."""
    result = runner.invoke(main, ["start"], input="")

    assert result.exit_code == 1
    output = _everything(result)
    assert "Error:" in output
    assert "No goal provided. Pass it as an argument:" in output
    assert 'furrow start "add a /health endpoint"' in output
    assert "Traceback" not in output


def test_a_blank_goal_argument_is_an_actionable_error(runner: CliRunner) -> None:
    result = runner.invoke(main, ["start", "   "])

    assert result.exit_code == 1
    assert "No goal provided." in _everything(result)


# --------------------------------------------------------------------------
# --model must set every role model, not just settings.model
# --------------------------------------------------------------------------


def test_model_sets_every_role_model(
    runner: CliRunner, global_settings: Any, started: dict[str, Any]
) -> None:
    """Regression guard: ``--model`` used to be a silent no-op.

    Every agent call passes an explicit role model, so setting ``settings.model``
    alone had no effect at all.
    """
    result = runner.invoke(main, ["start", "--model", "gpt-4o", "build a thing"])

    assert result.exit_code == 0, _everything(result)
    assert started["ran"] is True
    assert started["goal"] == "build a thing"
    for field in MODEL_FIELDS:
        assert getattr(global_settings, field) == "gpt-4o", field


def test_provider_openai_never_leaves_anthropic_model_ids_on_the_settings(
    runner: CliRunner, global_settings: Any, started: dict[str, Any]
) -> None:
    """Anthropic model ids answer 404 on OpenAI's API, so the defaults must swap."""
    result = runner.invoke(main, ["start", "--provider", "openai", "build a thing"])

    assert result.exit_code == 0, _everything(result)
    assert global_settings.provider == Provider.OPENAI
    assert global_settings.provider.value == "openai"
    assert global_settings.model == "gpt-4o"
    assert global_settings.planner_model == "gpt-4o-mini"
    assert global_settings.worker_model == "gpt-4o"
    assert global_settings.tester_model == "gpt-4o"
    for field in MODEL_FIELDS:
        assert not getattr(global_settings, field).startswith("claude-"), field


def test_an_explicit_worker_model_survives_a_provider_switch(
    runner: CliRunner, global_settings: Any, started: dict[str, Any]
) -> None:
    """Provider defaults only fill in what the user did not configure."""
    global_settings.worker_model = "my-tuned-worker"

    result = runner.invoke(main, ["start", "--provider", "openai", "build a thing"])

    assert result.exit_code == 0, _everything(result)
    assert global_settings.provider == Provider.OPENAI
    assert global_settings.worker_model == "my-tuned-worker"
    # The other three are still unset, so they take the provider defaults.
    assert global_settings.planner_model == PROVIDER_DEFAULT_MODELS["openai"]["planner_model"]
    assert global_settings.tester_model == PROVIDER_DEFAULT_MODELS["openai"]["tester_model"]


def test_an_unknown_provider_is_a_usage_error(
    runner: CliRunner, global_settings: Any, started: dict[str, Any]
) -> None:
    result = runner.invoke(main, ["start", "--provider", "bogus", "build a thing"])

    assert result.exit_code == 2
    assert "bogus" in _everything(result)
    assert started == {}, "the run must not start with an unusable provider"


# --------------------------------------------------------------------------
# Other flag translation
# --------------------------------------------------------------------------


def test_max_cycles_and_workspace_reach_the_settings(
    runner: CliRunner, global_settings: Any, started: dict[str, Any], tmp_path: Path
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()

    result = runner.invoke(
        main, ["start", "--max-cycles", "2", "--workspace", str(workspace), "build a thing"]
    )

    assert result.exit_code == 0, _everything(result)
    assert global_settings.max_cycles == 2
    assert global_settings.workspace == workspace


def test_verbose_selects_the_debug_log_level(
    runner: CliRunner, global_settings: Any, started: dict[str, Any]
) -> None:
    result = runner.invoke(main, ["start", "-v", "build a thing"])

    assert result.exit_code == 0, _everything(result)
    assert global_settings.log_level == "DEBUG"


# --------------------------------------------------------------------------
# web: forwarded kwargs
# --------------------------------------------------------------------------


def test_web_forwards_host_and_port(runner: CliRunner, served: dict[str, Any]) -> None:
    result = runner.invoke(main, ["web", "--host", "127.0.0.1", "--port", "9999"])

    assert result.exit_code == 0, _everything(result)
    assert served == {"host": "127.0.0.1", "port": 9999, "reload": False}


def test_web_reload_flag_and_defaults_are_forwarded(
    runner: CliRunner, served: dict[str, Any]
) -> None:
    result = runner.invoke(main, ["web", "--reload"])

    assert result.exit_code == 0, _everything(result)
    assert served == {"host": "127.0.0.1", "port": 8000, "reload": True}


def test_web_applies_the_model_flag_before_serving(
    runner: CliRunner, global_settings: Any, served: dict[str, Any]
) -> None:
    result = runner.invoke(main, ["web", "--model", "gpt-4o"])

    assert result.exit_code == 0, _everything(result)
    for field in MODEL_FIELDS:
        assert getattr(global_settings, field) == "gpt-4o", field
    assert served == {"host": "127.0.0.1", "port": 8000, "reload": False}


# --------------------------------------------------------------------------
# Failure modes: clean message, non-zero exit, no traceback
# --------------------------------------------------------------------------


def test_a_missing_api_key_fails_before_the_first_llm_call(
    runner: CliRunner, global_settings: Any, started: dict[str, Any], monkeypatch: Any
) -> None:
    """Regression guard: this used to print tracebacks and still exit 0.

    ``LLMClient`` resolves its key lazily and ``Orchestrator.run`` swallows any
    per-cycle exception, so the missing key surfaced three times as a traceback
    inside a run that reported success.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    global_settings.anthropic_api_key = None
    global_settings.provider = Provider.ANTHROPIC

    result = runner.invoke(main, ["start", "build a thing"])

    assert result.exit_code != 0
    output = _everything(result)
    assert "ANTHROPIC_API_KEY" in output
    assert "Traceback" not in output
    assert started == {}, "no orchestrator, so no LLM call and no credits burned"


def test_a_bare_env_api_key_satisfies_credentials(
    runner: CliRunner, global_settings: Any, started: dict[str, Any], monkeypatch: Any
) -> None:
    """False-positive guard: a bare provider env var alone is enough.

    ``_require_credentials`` mirrors ``LLMClient``'s two-source lookup -- the
    ``FURROW_*`` setting OR the bare env var. Setting only the env var (with the
    Settings field unset) must NOT trip the credential check and exit 0.
    """
    global_settings.anthropic_api_key = None
    global_settings.provider = Provider.ANTHROPIC
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")

    result = runner.invoke(main, ["start", "build a thing"])

    assert result.exit_code == 0, _everything(result)
    assert started["ran"] is True


def test_a_value_error_from_the_orchestrator_is_reported_cleanly(
    runner: CliRunner, global_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ValueError`` is an expected failure mode: one clean message, exit 1."""

    class _ExplodingOrchestrator:
        def __init__(self, goal: str, settings: Any = None) -> None:
            self.goal = goal

        async def run(self) -> None:
            raise ValueError("anthropic_api_key is required for the anthropic provider")

    monkeypatch.setattr("furrow.core.orchestrator.Orchestrator", _ExplodingOrchestrator)

    result = runner.invoke(main, ["start", "build a thing"])

    assert result.exit_code == 1
    output = _everything(result)
    assert "anthropic_api_key is required" in output
    assert "Traceback" not in output
    assert isinstance(result.exception, SystemExit)


def test_an_unexpected_exception_is_not_hidden_behind_a_clean_error(
    runner: CliRunner, global_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_user_errors`` handles only ``ValueError``/``ImportError``/``KeyboardInterrupt``.

    Anything else is a real bug and must still surface as one, so the traceback is
    never swallowed into a tidy CLI error.
    """

    class _BuggyOrchestrator:
        def __init__(self, goal: str, settings: Any = None) -> None:
            self.goal = goal

        async def run(self) -> None:
            raise AttributeError("NoneType has no attribute 'planner_model'")

    monkeypatch.setattr("furrow.core.orchestrator.Orchestrator", _BuggyOrchestrator)

    result = runner.invoke(main, ["start", "build a thing"])

    assert result.exit_code != 0
    assert isinstance(result.exception, AttributeError)
