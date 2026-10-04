"""The command line surface.

``click.testing.CliRunner`` drives the real group and its real options. Only the two
side effects that cannot happen inside a test are replaced: the orchestrator the ``start``
command runs, and the ``run()`` call the ``web`` command makes. Everything asserted here
is the option surface itself -- what exists, what it validates, and what it does to the
settings the agents receive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from furrow import __version__
from furrow.cli import main as cli_main
from furrow.cli.main import _effective_settings, main
from furrow.config import Provider, Settings, get_settings


@dataclass
class FakeOrchestrator:
    """Stand-in for the orchestrator ``start`` runs."""

    goal: str
    settings: Settings
    console: Any = None
    runs: list[str] = field(default_factory=list)
    interrupted: bool = False

    async def run(self) -> None:
        self.runs.append(self.goal)
        if self.interrupted:
            raise KeyboardInterrupt

    def cancel(self) -> None:
        self.interrupted = True


@dataclass
class WebCall:
    host: str
    port: int


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def fixed_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    """Give the CLI a deterministic settings object instead of the process singleton."""
    settings = Settings(workspace=tmp_path)
    monkeypatch.setattr(cli_main, "_settings", lambda: settings)
    return settings


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> list[FakeOrchestrator]:
    """Collect the orchestrators ``start`` builds instead of running a real loop."""
    import furrow.core.orchestrator as orchestrator_module

    built: list[FakeOrchestrator] = []

    def factory(**kwargs: Any) -> FakeOrchestrator:
        instance = FakeOrchestrator(**kwargs)
        built.append(instance)
        return instance

    monkeypatch.setattr(orchestrator_module, "Orchestrator", factory)
    return built


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[WebCall]:
    """Capture the host and port ``web`` would bind instead of starting a server."""
    import furrow.web.server as server_module

    calls: list[WebCall] = []
    monkeypatch.setattr(
        server_module, "run", lambda host=None, port=None: calls.append(WebCall(host, port))
    )
    return calls


# -------------------------------------------------------------------- version


def test_version_flag_prints_the_version(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"furrow {__version__}"


def test_group_help_documents_both_commands(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "start" in result.output
    assert "web" in result.output


# ------------------------------------------------------------- option surface


def test_start_exposes_every_documented_option(runner: CliRunner) -> None:
    result = runner.invoke(main, ["start", "--help"])
    assert result.exit_code == 0
    for option in ("--model", "--provider", "--workspace", "--max-cycles"):
        assert option in result.output


def test_web_exposes_every_documented_option(runner: CliRunner) -> None:
    result = runner.invoke(main, ["web", "--help"])
    assert result.exit_code == 0
    assert "--host" in result.output
    assert "--port" in result.output


def test_model_override_reaches_all_four_model_fields() -> None:
    settings = _effective_settings("some-model", None, None, None)
    assert settings.model == "some-model"
    assert settings.planner_model == "some-model"
    assert settings.worker_model == "some-model"
    assert settings.tester_model == "some-model"


def test_model_override_does_not_mutate_the_settings_singleton() -> None:
    before = get_settings().model_dump()
    _effective_settings("some-model", None, None, None)
    assert get_settings().model_dump() == before


def test_no_overrides_returns_the_settings_unchanged(fixed_settings: Settings) -> None:
    assert _effective_settings(None, None, None, None) is fixed_settings


def test_provider_override_is_applied() -> None:
    assert _effective_settings(None, "openai", None, None).provider is Provider.OPENAI


def test_provider_override_rejects_a_removed_provider(runner: CliRunner) -> None:
    result = runner.invoke(main, ["start", "goal", "--provider", "ollama"])
    assert result.exit_code == 2
    assert "ollama" in result.output
    assert "not a valid Provider" in result.output


def test_max_cycles_zero_is_allowed() -> None:
    assert _effective_settings(None, None, None, 0).max_cycles == 0


def test_max_cycles_override_is_applied() -> None:
    assert _effective_settings(None, None, None, 7).max_cycles == 7


def test_workspace_override_is_applied(tmp_path: Path) -> None:
    settings = _effective_settings(None, None, tmp_path, None)
    assert settings.workspace == tmp_path


# --------------------------------------------------------------------- start


def test_start_runs_the_goal_with_the_effective_settings(
    runner: CliRunner, started: list[FakeOrchestrator], tmp_path: Path
) -> None:
    result = runner.invoke(
        main,
        [
            "start",
            "add a widget",
            "--model",
            "some-model",
            "--provider",
            "anthropic",
            "--workspace",
            str(tmp_path),
            "--max-cycles",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert started[0].goal == "add a widget"
    assert started[0].runs == ["add a widget"]
    settings = started[0].settings
    assert (settings.model, settings.max_cycles, settings.workspace) == ("some-model", 2, tmp_path)
    assert settings.provider is Provider.ANTHROPIC


def test_start_reports_the_effective_configuration(
    runner: CliRunner, started: list[FakeOrchestrator]
) -> None:
    result = runner.invoke(main, ["start", "goal", "--model", "shown-model"])
    assert result.exit_code == 0, result.output
    assert "model=shown-model" in result.output


def test_start_prompts_for_a_goal_when_none_is_given(
    runner: CliRunner, started: list[FakeOrchestrator]
) -> None:
    result = runner.invoke(main, ["start"], input="build a parser\n")
    assert result.exit_code == 0, result.output
    assert started[0].goal == "build a parser"


def test_start_aborts_when_the_goal_prompt_is_answered_with_nothing(
    runner: CliRunner, started: list[FakeOrchestrator]
) -> None:
    result = runner.invoke(main, ["start"], input="")
    assert result.exit_code == 1
    assert started == []


def test_start_refuses_a_whitespace_goal(
    runner: CliRunner, started: list[FakeOrchestrator]
) -> None:
    result = runner.invoke(main, ["start", "   "])
    assert result.exit_code == 2
    assert "goal must not be empty" in result.output
    assert started == []


def test_start_refuses_a_workspace_that_does_not_exist(
    runner: CliRunner, started: list[FakeOrchestrator]
) -> None:
    result = runner.invoke(main, ["start", "goal", "--workspace", "/no/such/place"])
    assert result.exit_code == 2
    assert "existing directory" in result.output
    assert started == []


def test_start_refuses_a_negative_cycle_limit(runner: CliRunner) -> None:
    result = runner.invoke(main, ["start", "goal", "--max-cycles", "-1"])
    assert result.exit_code == 2


def test_start_exits_130_when_interrupted(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import furrow.core.orchestrator as orchestrator_module

    def interrupting(**kwargs: Any) -> FakeOrchestrator:
        instance = FakeOrchestrator(**kwargs)
        instance.interrupted = True
        return instance

    monkeypatch.setattr(orchestrator_module, "Orchestrator", interrupting)
    result = runner.invoke(main, ["start", "goal"])
    assert result.exit_code == 130
    assert "Furrow stopped by user" in result.output


def test_start_honours_the_configured_log_level(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    started: list[FakeOrchestrator],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        cli_main,
        "_settings",
        lambda: Settings(workspace=tmp_path, log_level="DEBUG"),
    )
    root_level = logging.getLogger().level
    try:
        result = runner.invoke(main, ["start", "goal"])
        assert result.exit_code == 0, result.output
        assert logging.getLogger().level == logging.DEBUG
    finally:
        logging.getLogger().setLevel(root_level)


# ----------------------------------------------------------------------- web


def test_web_binds_the_requested_host_and_port(runner: CliRunner, served: list[WebCall]) -> None:
    result = runner.invoke(main, ["web", "--host", "localhost", "--port", "8123"])
    assert result.exit_code == 0, result.output
    assert served == [WebCall("localhost", 8123)]


def test_web_falls_back_to_the_configured_defaults(
    runner: CliRunner, served: list[WebCall]
) -> None:
    result = runner.invoke(main, ["web"])
    assert result.exit_code == 0, result.output
    assert served == [WebCall("127.0.0.1", 8000)]


@pytest.mark.parametrize("port", ["0", "65536", "-1"])
def test_web_refuses_an_out_of_range_port(runner: CliRunner, port: str) -> None:
    result = runner.invoke(main, ["web", "--port", port])
    assert result.exit_code == 2
    assert "between 1 and 65535" in result.output


def test_web_exits_130_when_interrupted(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    import furrow.web.server as server_module

    def interrupting(host: str | None = None, port: int | None = None) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(server_module, "run", interrupting)
    result = runner.invoke(main, ["web"])
    assert result.exit_code == 130
    assert "Web server stopped by user" in result.output


def test_help_needs_no_credentials(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    import furrow.config as config_module

    def exploding_settings() -> Settings:
        raise AssertionError("--help must not build a Settings object")

    monkeypatch.setattr(cli_main, "_settings", exploding_settings)
    monkeypatch.setattr(config_module, "get_settings", exploding_settings)
    assert runner.invoke(main, ["--help"]).exit_code == 0
