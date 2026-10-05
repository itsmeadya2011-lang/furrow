"""Regression tests for the command line entry point (``furrow.cli.main``).

``click.testing.CliRunner`` drives the real commands; only ``LLMClient.complete``
is stubbed, and ``--workspace``/``--test-command`` keep every run inside
``tmp_path`` with no subprocess and no network access.

Load-bearing regressions pinned here:

* ``--model`` reaches the **planner, worker and tester**, not just
  ``settings.model``, and it beats ``--planner-model``;
* an invalid option value is a friendly usage error with no traceback;
* the exit codes are 0 (completed), non-zero (halted) and 130 (interrupted);
* ``configure_logging`` runs for both ``start`` and ``web``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from furrow.cli import main as cli
from furrow.llm import LLMClient
from tests.conftest import (
    PLANNER,
    TESTER,
    WORKER,
    ScriptedCompletion,
    install_scripted,
    plan_reply,
    task,
    verdict_json,
    worker_reply,
)

PASSING = verdict_json(True, "all green")
FAILING = verdict_json(False, "one failed", ["boom"])
PLANNER_REPLY = plan_reply(task("1", "make a module"))
WORKER_REPLY = worker_reply("wrote it")


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def base_args(workspace: Path) -> list[str]:
    return ["--workspace", str(workspace), "--test-command", "true"]


def happy_llm(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> ScriptedCompletion:
    scripted = install_scripted(monkeypatch, planner=PLANNER_REPLY, worker=WORKER_REPLY)
    scripted._replies[TESTER] = PASSING
    return scripted


def assert_no_traceback(result: Any) -> None:
    assert result.exit_code != 0
    assert "Traceback (most recent call last)" not in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


# ------------------------------------------------------------- meta options


def test_version_exits_zero(runner: CliRunner) -> None:
    result = runner.invoke(cli.main, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip().startswith("furrow")


def test_help_exits_zero(runner: CliRunner) -> None:
    result = runner.invoke(cli.main, ["--help"])
    assert result.exit_code == 0
    assert "start" in result.output
    assert "web" in result.output


def test_start_help_exits_zero(runner: CliRunner) -> None:
    result = runner.invoke(cli.main, ["start", "--help"])
    assert result.exit_code == 0
    for option in ("--model", "--planner-model", "--worker-model", "--tester-model", "--workspace"):
        assert option in result.output


def test_web_help_exits_zero(runner: CliRunner) -> None:
    result = runner.invoke(cli.main, ["web", "--help"])
    assert result.exit_code == 0
    assert "--host" in result.output
    assert "--port" in result.output


def test_no_arguments_prints_the_usage_help(runner: CliRunner) -> None:
    """Bare ``furrow`` is click's "missing command" usage error, not a traceback."""
    result = runner.invoke(cli.main, [])
    assert result.exit_code == 2
    assert "Traceback (most recent call last)" not in result.output
    assert "Usage:" in result.output
    assert "start" in result.output


def test_an_unknown_command_is_a_usage_error(runner: CliRunner) -> None:
    result = runner.invoke(cli.main, ["frobnicate"])
    assert_no_traceback(result)


# ------------------------------------------------------------- model wiring


def test_model_reaches_the_planner_the_worker_and_the_tester(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripted = happy_llm(monkeypatch)
    result = runner.invoke(
        cli.main, ["start", "goal", *base_args(tmp_path), "--model", "shared-model"]
    )
    assert result.exit_code == 0
    assert scripted.models(PLANNER) == ["shared-model"]
    assert scripted.models(WORKER) == ["shared-model"]
    assert scripted.models(TESTER) == ["shared-model"]


def test_model_beats_planner_model(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripted = happy_llm(monkeypatch)
    result = runner.invoke(
        cli.main,
        [
            "start",
            "goal",
            *base_args(tmp_path),
            "--model",
            "shared-model",
            "--planner-model",
            "planner-only",
        ],
    )
    assert result.exit_code == 0
    assert scripted.models(PLANNER) == ["shared-model"]


def test_model_beats_every_per_agent_option(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripted = happy_llm(monkeypatch)
    result = runner.invoke(
        cli.main,
        [
            "start",
            "goal",
            *base_args(tmp_path),
            "--planner-model",
            "p",
            "--worker-model",
            "w",
            "--tester-model",
            "t",
            "--model",
            "shared-model",
        ],
    )
    assert result.exit_code == 0
    assert scripted.models(PLANNER) == ["shared-model"]
    assert scripted.models(WORKER) == ["shared-model"]
    assert scripted.models(TESTER) == ["shared-model"]


def test_the_per_agent_options_apply_on_their_own(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripted = happy_llm(monkeypatch)
    result = runner.invoke(
        cli.main,
        [
            "start",
            "goal",
            *base_args(tmp_path),
            "--planner-model",
            "p",
            "--worker-model",
            "w",
            "--tester-model",
            "t",
        ],
    )
    assert result.exit_code == 0
    assert scripted.models(PLANNER) == ["p"]
    assert scripted.models(WORKER) == ["w"]
    assert scripted.models(TESTER) == ["t"]


def test_the_model_overrides_are_resolved_without_a_client(
    runner: CliRunner, tmp_path: Path
) -> None:
    resolved = cli._model_overrides("m", "p", "w", "t")
    assert resolved == {
        "model": "m",
        "planner_model": "m",
        "worker_model": "m",
        "tester_model": "m",
    }
    assert cli._model_overrides(None, "p", None, "t") == {
        "planner_model": "p",
        "tester_model": "t",
    }
    assert cli._model_overrides(None, None, None, None) == {}


# ------------------------------------------------------- invalid options


@pytest.mark.parametrize(
    ("options", "needle"),
    [
        (["--max-parallel-tasks", "0"], "max_parallel_tasks"),
        (["--max-parallel-tasks", "-2"], "max_parallel_tasks"),
        (["--max-cycles", "-1"], "max_cycles"),
        (["--log-level", "verbose"], "--log-level"),
        (["--model", "   "], "model"),
        (["--provider", "bedrock"], "provider"),
        (["--test-timeout", "0"], "test-timeout"),
    ],
)
def test_an_invalid_option_value_is_a_friendly_error(
    runner: CliRunner, tmp_path: Path, options: list[str], needle: str
) -> None:
    result = runner.invoke(cli.main, ["start", "goal", *base_args(tmp_path), *options])
    assert_no_traceback(result)
    assert "Error" in result.output
    assert needle in result.output


def test_a_whitespace_model_is_refused_without_running(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripted = happy_llm(monkeypatch)
    result = runner.invoke(cli.main, ["start", "goal", *base_args(tmp_path), "--model", "  "])
    assert_no_traceback(result)
    assert scripted.calls == []


def test_build_settings_reports_a_validation_error_as_a_usage_error() -> None:
    import click

    with pytest.raises(click.UsageError):
        cli._build_settings(max_parallel_tasks=0)


# --------------------------------------------------------------- exit codes


def test_a_completed_run_exits_zero(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    happy_llm(monkeypatch)
    result = runner.invoke(cli.main, ["start", "goal", *base_args(tmp_path)])
    assert result.exit_code == cli.EXIT_COMPLETED
    assert result.exit_code == 0


def test_a_halted_run_exits_non_zero(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripted = install_scripted(
        monkeypatch, planner=PLANNER_REPLY, worker=WORKER_REPLY, tester=FAILING
    )
    result = runner.invoke(cli.main, ["start", "goal", *base_args(tmp_path), "--max-cycles", "2"])
    assert result.exit_code == cli.EXIT_HALTED
    assert result.exit_code != 0
    assert len(scripted.by_role[PLANNER]) == 2


def test_a_planner_failure_exits_non_zero(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_scripted(monkeypatch, planner="I will not plan that.")
    result = runner.invoke(cli.main, ["start", "goal", *base_args(tmp_path)])
    assert result.exit_code == cli.EXIT_HALTED


def test_an_interrupt_exits_130(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    happy_llm(monkeypatch)

    def interrupt(coro: Any) -> Any:
        coro.close()
        raise KeyboardInterrupt

    monkeypatch.setattr(asyncio, "run", interrupt)
    result = runner.invoke(cli.main, ["start", "goal", *base_args(tmp_path)])
    assert result.exit_code == cli.EXIT_INTERRUPTED
    assert result.exit_code == 130
    assert "Interrupted" in result.output


def test_an_empty_prompted_goal_exits_non_zero(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    happy_llm(monkeypatch)
    result = runner.invoke(cli.main, ["start", *base_args(tmp_path)], input="\n\n")
    assert result.exit_code != 0


def test_a_missing_goal_is_prompted_for(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripted = happy_llm(monkeypatch)
    result = runner.invoke(cli.main, ["start", *base_args(tmp_path)], input="typed goal\n")
    assert result.exit_code == 0
    assert "typed goal" in scripted.prompts(PLANNER)[0]


# ---------------------------------------------------------- configure_logging


def test_start_configures_logging(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    happy_llm(monkeypatch)
    seen: list[str | None] = []
    monkeypatch.setattr(cli, "configure_logging", lambda level=None: seen.append(level))
    result = runner.invoke(cli.main, ["start", "goal", *base_args(tmp_path)])
    assert result.exit_code == 0
    assert seen == ["INFO"]


def test_start_configures_logging_with_the_requested_level(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    happy_llm(monkeypatch)
    seen: list[str | None] = []
    monkeypatch.setattr(cli, "configure_logging", lambda level=None: seen.append(level))
    result = runner.invoke(
        cli.main, ["start", "goal", *base_args(tmp_path), "--log-level", "debug"]
    )
    assert result.exit_code == 0
    assert seen == ["DEBUG"]


def test_web_configures_logging_and_starts_the_server(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    from furrow.web import server

    seen: list[str | None] = []
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(cli, "configure_logging", lambda level=None: seen.append(level))
    monkeypatch.setattr(server, "run", lambda **kwargs: calls.append(kwargs))
    result = runner.invoke(cli.main, ["web", "--port", "9999", "--log-level", "DEBUG"])
    assert result.exit_code == 0
    assert seen == ["DEBUG"]
    assert calls[0]["host"] == "127.0.0.1"
    assert calls[0]["port"] == 9999
    assert calls[0]["log_level"] == "DEBUG"


def test_web_passes_the_cycle_budget_and_the_concurrency(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    from furrow.web import server

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(cli, "configure_logging", lambda level=None: None)
    monkeypatch.setattr(server, "run", lambda **kwargs: calls.append(kwargs))
    result = runner.invoke(cli.main, ["web", "--max-cycles", "3", "--max-concurrent-runs", "1"])
    assert result.exit_code == 0
    assert calls[0]["max_concurrent_runs"] == 1
    assert calls[0]["settings"].max_cycles == 3


@pytest.mark.parametrize(
    ("options", "needle"),
    [
        (["--port", "0"], "--port"),
        (["--port", "70000"], "--port"),
        (["--max-cycles", "-1"], "--max-cycles"),
        (["--max-concurrent-runs", "0"], "--max-concurrent-runs"),
        (["--log-level", "nope"], "--log-level"),
    ],
)
def test_web_rejects_invalid_options(runner: CliRunner, options: list[str], needle: str) -> None:
    result = runner.invoke(cli.main, ["web", *options])
    assert_no_traceback(result)
    assert needle in result.output


def test_web_warns_when_exposed_without_a_token(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    from furrow.web import server

    monkeypatch.setattr(cli, "configure_logging", lambda level=None: None)
    monkeypatch.setattr(server, "run", lambda **kwargs: None)
    monkeypatch.delenv(server.TOKEN_ENV, raising=False)
    result = runner.invoke(cli.main, ["web", "--host", "0.0.0.0"])
    assert result.exit_code == 0
    assert "warning" in result.output.lower()


# -------------------------------------------------------------- internals


def test_package_version_is_a_string() -> None:
    assert isinstance(cli.package_version(), str)


def test_the_exit_code_constants_are_the_documented_ones() -> None:
    assert cli.EXIT_COMPLETED == 0
    assert cli.EXIT_HALTED == 1
    assert cli.EXIT_INTERRUPTED == 130
    assert "127.0.0.1" in cli.LOOPBACK_HOSTS


def test_run_goal_closes_the_client(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    happy_llm(monkeypatch)
    closed: list[bool] = []
    original = LLMClient.aclose

    async def tracked(self: LLMClient) -> None:
        closed.append(True)
        await original(self)

    monkeypatch.setattr(LLMClient, "aclose", tracked)
    result = runner.invoke(cli.main, ["start", "goal", *base_args(tmp_path)])
    assert result.exit_code == 0
    assert closed, "the httpx client must be released"
