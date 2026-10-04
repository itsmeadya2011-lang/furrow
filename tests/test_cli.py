"""Tests for :mod:`furrow.cli.main`: option validation, exit codes, wiring.

Every command that would start a run is intercepted with a fake orchestrator, so
``start`` exits immediately and deterministically without contacting a provider.
The ``get_settings()`` singleton the CLI mutates is restored by the autouse
fixture in ``conftest.py``. ``start`` also calls ``configure_logging``, which
mutates the process-wide root logger, so the module-local
:func:`_isolate_root_logging` fixture restores that in turn.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import pytest
from click.testing import CliRunner

import furrow.cli.main as cli_module
import furrow.config as config_module
import furrow.core.orchestrator as orchestrator_module
from furrow import __version__
from furrow.cli.main import main
from furrow.config import LOG_LEVELS, Provider, Settings, TestResult
from furrow.core.events import RunResult

COMPLETED_RESULT = RunResult(
    cycles=1,
    completed=True,
    tasks_completed=1,
    stopped_reason="All 1 planned tasks completed and the test suite passed.",
)

INCOMPLETE_RESULT = RunResult(
    cycles=1,
    completed=False,
    tasks_completed=0,
    tasks_failed=1,
    last_error="RuntimeError: worker exploded",
    stopped_reason="Run aborted by an unexpected error.",
)


class FakeOrchestrator:
    """Stands in for :class:`~furrow.core.orchestrator.Orchestrator` in the CLI."""

    #: Shared state describing what the most recent construction saw.
    captured: dict[str, Any] = {}
    #: The result (or exception) the next ``run`` should produce.
    outcome: Any = COMPLETED_RESULT

    def __init__(
        self,
        goal: str,
        client: Any = None,
        settings: Any = None,
        on_event: Any = None,
    ) -> None:
        """Record the constructor arguments.

        Args:
            goal: The resolved goal.
            client: Unused; the CLI builds none.
            settings: The resolved settings singleton.
            on_event: The event reporter, or ``None`` under ``--quiet``.
        """
        type(self).captured = {
            "goal": goal,
            "settings": settings,
            "on_event": on_event,
        }

    async def run(self) -> RunResult:
        """Return or raise the scripted outcome.

        Returns:
            ``type(self).outcome`` when it is a result.

        Raises:
            BaseException: ``type(self).outcome`` when it is an exception.
        """
        outcome = type(self).outcome
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture
def runner() -> CliRunner:
    """Return a :class:`~click.testing.CliRunner` with mix_stderr captured.

    Returns:
        A runner that keeps stderr out of ``output`` unless a test asks for it.
    """
    return CliRunner()


@pytest.fixture
def fake_orchestrator(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Replace the orchestrator the CLI imports inside ``start``.

    Args:
        monkeypatch: Pytest's patcher.

    Returns:
        The :class:`FakeOrchestrator` class, so a test can set ``outcome``.
    """
    FakeOrchestrator.captured = {}
    FakeOrchestrator.outcome = COMPLETED_RESULT
    monkeypatch.setattr(orchestrator_module, "Orchestrator", FakeOrchestrator)
    return FakeOrchestrator


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make an Anthropic key resolvable so ``start`` gets past validation.

    Args:
        monkeypatch: Pytest's patcher.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def _isolate_root_logging() -> Iterator[None]:
    """Snapshot and restore the root logger, which ``start`` re-levels.

    ``furrow start`` calls ``configure_logging`` on every invocation, so without
    this the level applied by one CLI test would still be installed when the
    next module ran.

    Yields:
        None, for tests that only need the state snapshotted.
    """
    root = logging.getLogger()
    saved_level = root.level
    saved_handlers = list(root.handlers)
    try:
        yield
    finally:
        root.setLevel(saved_level)
        root.handlers[:] = saved_handlers


# --------------------------------------------------------------------------- #
# group metadata
# --------------------------------------------------------------------------- #


def test_help_lists_both_commands(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "start" in result.output
    assert "web" in result.output


def test_version_prints_the_package_version(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.output


def test_start_help_documents_the_model_flag(runner: CliRunner) -> None:
    result = runner.invoke(main, ["start", "--help"])

    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--max-parallel" in result.output


# --------------------------------------------------------------------------- #
# the --model regression
# --------------------------------------------------------------------------- #


def test_model_flag_sets_override_and_leaves_model_untouched(
    runner: CliRunner, fake_orchestrator: Any, api_key: None
) -> None:
    """``--model`` used to write ``settings.model``, where nothing read it."""
    original_model = config_module.get_settings().model

    result = runner.invoke(main, ["start", "ship it", "--model", "gpt-4o-mini", "--quiet"])

    assert result.exit_code == 0
    settings = config_module.get_settings()
    assert settings.override_model == "gpt-4o-mini"
    assert settings.model == original_model


def test_model_flag_reaches_the_banner_and_the_orchestrator(
    runner: CliRunner, fake_orchestrator: Any, api_key: None
) -> None:
    result = runner.invoke(main, ["start", "ship it", "--model", "gpt-4o-mini"])

    assert result.exit_code == 0
    assert "gpt-4o-mini" in result.output
    assert fake_orchestrator.captured["settings"].override_model == "gpt-4o-mini"


def test_without_the_model_flag_override_stays_none(
    runner: CliRunner, fake_orchestrator: Any, api_key: None
) -> None:
    result = runner.invoke(main, ["start", "ship it", "--quiet"])

    assert result.exit_code == 0
    assert config_module.get_settings().override_model is None


def test_override_model_beats_every_role_specific_model(
    runner: CliRunner, fake_orchestrator: Any, api_key: None
) -> None:
    """The resolution order every agent depends on: override, then role, then default."""
    runner.invoke(
        main, ["start", "ship it", "--model", "override", "--quiet"]
    )

    settings = config_module.get_settings()

    assert settings.override_model == "override"
    assert settings.planner_model != "override"
    assert settings.worker_model != "override"
    assert settings.tester_model != "override"
    assert settings.model != "override"


# --------------------------------------------------------------------------- #
# option validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value", ["0", "-1", "-100"])
def test_max_parallel_below_one_is_rejected(
    runner: CliRunner, value: str, fake_orchestrator: Any, api_key: None
) -> None:
    result = runner.invoke(main, ["start", "ship it", "--max-parallel", value])

    assert result.exit_code == 2
    assert "max-parallel" in result.output
    assert fake_orchestrator.captured == {}


def test_max_parallel_one_is_accepted(
    runner: CliRunner, fake_orchestrator: Any, api_key: None
) -> None:
    result = runner.invoke(main, ["start", "ship it", "--max-parallel", "1", "--quiet"])

    assert result.exit_code == 0
    assert config_module.get_settings().max_parallel_tasks == 1


def test_negative_max_cycles_is_rejected(
    runner: CliRunner, fake_orchestrator: Any, api_key: None
) -> None:
    result = runner.invoke(main, ["start", "ship it", "--max-cycles", "-1"])

    assert result.exit_code == 2
    assert "max-cycles" in result.output
    assert fake_orchestrator.captured == {}


def test_zero_max_cycles_is_accepted_as_unlimited(
    runner: CliRunner, fake_orchestrator: Any, api_key: None
) -> None:
    result = runner.invoke(main, ["start", "ship it", "--max-cycles", "0", "--quiet"])

    assert result.exit_code == 0
    assert config_module.get_settings().max_cycles == 0


def test_a_missing_workspace_directory_is_rejected(
    runner: CliRunner, fake_orchestrator: Any, api_key: None
) -> None:
    result = runner.invoke(main, ["start", "ship it", "--workspace", "/no/such/dir"])

    assert result.exit_code == 2
    assert "workspace" in result.output


def test_options_are_written_into_the_settings_singleton(
    runner: CliRunner, fake_orchestrator: Any, api_key: None, tmp_path: Any
) -> None:
    result = runner.invoke(
        main,
        [
            "start",
            "ship it",
            "--max-parallel",
            "3",
            "--max-cycles",
            "5",
            "--test-command",
            "make check",
            "--workspace",
            str(tmp_path),
            "--quiet",
        ],
    )

    assert result.exit_code == 0
    settings = config_module.get_settings()
    assert settings.max_parallel_tasks == 3
    assert settings.max_cycles == 5
    assert settings.test_command == "make check"
    assert settings.workspace == tmp_path.resolve()


def test_provider_flag_is_validated(runner: CliRunner, api_key: None) -> None:
    result = runner.invoke(main, ["start", "ship it", "--provider", "not-a-provider"])

    assert result.exit_code == 2


# --------------------------------------------------------------------------- #
# --log-level
# --------------------------------------------------------------------------- #


def test_start_help_documents_the_log_level_flag(runner: CliRunner) -> None:
    result = runner.invoke(main, ["start", "--help"])
    choices = f"[{'|'.join(level.lower() for level in LOG_LEVELS)}]"

    assert result.exit_code == 0
    assert "--log-level" in result.output
    assert choices in result.output


def test_log_level_flag_applies_the_requested_level(
    runner: CliRunner, fake_orchestrator: Any, api_key: None
) -> None:
    result = runner.invoke(main, ["start", "ship it", "--log-level", "debug", "--quiet"])

    assert result.exit_code == 0
    assert config_module.get_settings().log_level == "DEBUG"
    assert logging.getLogger().level == logging.DEBUG


def test_log_level_flag_is_case_insensitive(
    runner: CliRunner, fake_orchestrator: Any, api_key: None
) -> None:
    result = runner.invoke(main, ["start", "ship it", "--log-level", "CrItIcAl", "--quiet"])

    assert result.exit_code == 0
    assert config_module.get_settings().log_level == "CRITICAL"
    assert logging.getLogger().level == logging.CRITICAL


def test_without_the_flag_the_configured_level_is_applied(
    runner: CliRunner, fake_orchestrator: Any, api_key: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FURROW_LOG_LEVEL", "warning")

    result = runner.invoke(main, ["start", "ship it", "--quiet"])

    assert result.exit_code == 0
    assert config_module.get_settings().log_level == "WARNING"
    assert logging.getLogger().level == logging.WARNING


def test_an_unknown_log_level_is_rejected_before_any_work(
    runner: CliRunner, fake_orchestrator: Any, api_key: None
) -> None:
    result = runner.invoke(main, ["start", "ship it", "--log-level", "trace"])

    assert result.exit_code == 2
    assert "--log-level" in result.output
    assert fake_orchestrator.captured == {}


def test_the_log_level_reaches_the_orchestrator_settings(
    runner: CliRunner, fake_orchestrator: Any, api_key: None
) -> None:
    result = runner.invoke(main, ["start", "ship it", "--log-level", "ERROR", "--quiet"])

    assert result.exit_code == 0
    assert fake_orchestrator.captured["settings"].log_level == "ERROR"


def test_the_level_is_live_before_the_orchestrator_runs(
    runner: CliRunner, fake_orchestrator: Any, api_key: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``configure_logging`` runs before the loop, so its first log line is visible."""
    observed: dict[str, int] = {}
    real_init = FakeOrchestrator.__init__

    def recording_init(self: Any, *args: Any, **kwargs: Any) -> None:
        observed["root_level"] = logging.getLogger().level
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(FakeOrchestrator, "__init__", recording_init)

    result = runner.invoke(main, ["start", "ship it", "--log-level", "debug", "--quiet"])

    assert result.exit_code == 0
    assert observed["root_level"] == logging.DEBUG


def test_a_rejected_log_level_leaves_the_root_logger_alone(
    runner: CliRunner, fake_orchestrator: Any, api_key: None
) -> None:
    """Click validates the choice before the command body, so nothing is applied."""
    before = logging.getLogger().level

    result = runner.invoke(main, ["start", "ship it", "--log-level", "trace"])

    assert result.exit_code == 2
    assert logging.getLogger().level == before


# --------------------------------------------------------------------------- #
# credential checks
# --------------------------------------------------------------------------- #


def test_a_missing_anthropic_key_fails_fast(
    runner: CliRunner, fake_orchestrator: Any
) -> None:
    result = runner.invoke(main, ["start", "ship it"])

    assert result.exit_code == 2
    assert "ANTHROPIC_API_KEY" in result.output
    assert fake_orchestrator.captured == {}


def test_a_missing_openai_key_fails_fast(runner: CliRunner, fake_orchestrator: Any) -> None:
    result = runner.invoke(main, ["start", "ship it", "--provider", "openai"])

    assert result.exit_code == 2
    assert "OPENAI_API_KEY" in result.output
    assert fake_orchestrator.captured == {}


def test_ollama_needs_no_api_key(runner: CliRunner, fake_orchestrator: Any) -> None:
    result = runner.invoke(
        main, ["start", "ship it", "--provider", "ollama", "--quiet"]
    )

    assert result.exit_code == 0
    assert config_module.get_settings().provider is Provider.OLLAMA


# --------------------------------------------------------------------------- #
# exit codes
# --------------------------------------------------------------------------- #


def test_a_completed_run_exits_zero(
    runner: CliRunner, fake_orchestrator: Any, api_key: None
) -> None:
    fake_orchestrator.outcome = COMPLETED_RESULT

    result = runner.invoke(main, ["start", "ship it"])

    assert result.exit_code == 0
    assert "Goal complete." in result.output


def test_an_incomplete_run_exits_one(
    runner: CliRunner, fake_orchestrator: Any, api_key: None
) -> None:
    fake_orchestrator.outcome = INCOMPLETE_RESULT

    result = runner.invoke(main, ["start", "ship it"])

    assert result.exit_code == 1
    assert "without completing the goal" in result.output
    assert "RuntimeError: worker exploded" in result.output


def test_a_failing_test_suite_exits_one_even_with_every_task_done(
    runner: CliRunner, fake_orchestrator: Any, api_key: None
) -> None:
    """``RunResult.completed`` already encodes the red suite; the CLI trusts it."""
    fake_orchestrator.outcome = RunResult(
        cycles=1,
        completed=False,
        tasks_completed=2,
        stopped_reason="Reached the configured max_cycles limit of 1.",
    )

    result = runner.invoke(main, ["start", "ship it"])

    assert result.exit_code == 1


def test_a_keyboard_interrupt_exits_130(
    runner: CliRunner, fake_orchestrator: Any, api_key: None
) -> None:
    fake_orchestrator.outcome = KeyboardInterrupt()

    result = runner.invoke(main, ["start", "ship it"])

    assert result.exit_code == 130
    assert "stopped by user" in result.output


def test_an_unexpected_error_exits_one(
    runner: CliRunner, fake_orchestrator: Any, api_key: None
) -> None:
    fake_orchestrator.outcome = RuntimeError("kaboom")

    result = runner.invoke(main, ["start", "ship it"])

    assert result.exit_code == 1
    assert "kaboom" in result.output


def test_the_orchestrator_receives_the_shared_singleton(
    runner: CliRunner, fake_orchestrator: Any, api_key: None
) -> None:
    result = runner.invoke(main, ["start", "ship it", "--quiet"])

    assert result.exit_code == 0
    assert fake_orchestrator.captured["settings"] is config_module.get_settings()
    assert fake_orchestrator.captured["goal"] == "ship it"


def test_quiet_suppresses_the_banner_and_the_event_reporter(
    runner: CliRunner, fake_orchestrator: Any, api_key: None
) -> None:
    result = runner.invoke(main, ["start", "ship it", "--quiet"])

    assert result.exit_code == 0
    assert fake_orchestrator.captured["on_event"] is None
    assert "Parallelism:" not in result.output
    assert "Furrow is working" not in result.output


def test_the_banner_reports_the_resolved_configuration(
    runner: CliRunner, fake_orchestrator: Any, api_key: None
) -> None:
    result = runner.invoke(main, ["start", "ship it", "--max-parallel", "3"])

    assert result.exit_code == 0
    assert "Parallelism:" in result.output
    assert "anthropic" in result.output


def test_a_blank_goal_is_prompted_for(
    runner: CliRunner, fake_orchestrator: Any, api_key: None
) -> None:
    result = runner.invoke(main, ["start"], input="typed goal\n")

    assert result.exit_code == 0
    assert fake_orchestrator.captured["goal"] == "typed goal"


# --------------------------------------------------------------------------- #
# web command
# --------------------------------------------------------------------------- #


def test_web_forwards_host_port_and_reload(runner: CliRunner) -> None:
    import furrow.web.server as web_module

    recorded: dict[str, Any] = {}
    original = web_module.run
    web_module.run = lambda **kwargs: recorded.update(kwargs)
    try:
        result = runner.invoke(main, ["web", "--host", "0.0.0.0", "--port", "9999", "--reload"])
    finally:
        web_module.run = original

    assert result.exit_code == 0
    assert recorded == {"host": "0.0.0.0", "port": 9999, "reload": True}


def test_web_defaults_to_loopback(runner: CliRunner) -> None:
    import furrow.web.server as web_module

    recorded: dict[str, Any] = {}
    original = web_module.run
    web_module.run = lambda **kwargs: recorded.update(kwargs)
    try:
        result = runner.invoke(main, ["web"])
    finally:
        web_module.run = original

    assert result.exit_code == 0
    assert recorded == {"host": "127.0.0.1", "port": 8000, "reload": False}


# --------------------------------------------------------------------------- #
# unit-level checks of the helpers
# --------------------------------------------------------------------------- #


def test_validate_options_accepts_none() -> None:
    assert cli_module._validate_options(None, None, None) is None


def test_validate_options_rejects_a_bad_workspace() -> None:
    import click

    with pytest.raises(click.BadParameter):
        cli_module._validate_options(None, None, "/no/such/dir")


def test_configure_aborts_without_a_key(tmp_path: Any) -> None:
    settings = Settings(_env_file=None, workspace=tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        cli_module._configure(settings, None, None, None, None, None, None)

    assert excinfo.value.code == cli_module.EXIT_CONFIG


def test_configure_returns_the_effective_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    settings = Settings(_env_file=None, anthropic_api_key="k")

    provider, model = cli_module._configure(
        settings, None, "chosen", None, None, None, None
    )

    assert provider is Provider.ANTHROPIC
    assert model == "chosen"
    assert settings.override_model == "chosen"


def test_exit_code_constants() -> None:
    assert cli_module.EXIT_OK == 0
    assert cli_module.EXIT_FAILED == 1
    assert cli_module.EXIT_CONFIG == 2
    assert cli_module.EXIT_INTERRUPTED == 130


def test_test_result_is_not_collected_as_a_test() -> None:
    assert TestResult.__test__ is False
