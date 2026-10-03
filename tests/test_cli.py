"""CLI tests: hermetic, no network, no LLM call, no server, no real credentials.

Every test drives the click group through ``CliRunner`` with ``Orchestrator`` and
``furrow.web.server.run`` replaced by recorders, so the only thing exercised is
the CLI's own behaviour: flag plumbing into a per-invocation ``Settings``, the
interactive prompt, exit codes and error messages.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result

import furrow.cli.main as cli
import furrow.config
import furrow.web.server
from furrow.cli.main import WEB_HOST, WEB_PORT, build_settings, get_version, main
from furrow.config import Provider, Settings

REPO_ROOT = Path(__file__).resolve().parents[1]

KEY_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "FURROW_ANTHROPIC_API_KEY",
    "FURROW_OPENAI_API_KEY",
)
FURROW_ENV_PREFIXES = ("FURROW_",)
#: Never sent anywhere: the orchestrator is faked, this only gets the CLI past its
#: own up-front credential check.
FAKE_ANTHROPIC_KEY = "sk-furrow-cli-test-not-a-real-key"
FAKE_OPENAI_KEY = "sk-furrow-cli-test-openai-not-real"


class FakeOrchestrator:
    """Records what the CLI passed and runs nothing."""

    calls: list[dict[str, Any]] = []
    #: set by a test to make ``run()`` raise instead of returning
    raises: BaseException | None = None

    def __init__(self, goal: str, client: Any = None, settings: Settings | None = None, **_: Any):
        self.goal = goal
        self.client = client
        self.settings = settings
        self.cycles = 0
        type(self).calls.append({"goal": goal, "settings": settings, "client": client})

    async def run(self) -> None:
        if type(self).raises is not None:
            raise type(self).raises

    @classmethod
    def reset(cls) -> None:
        cls.calls = []
        cls.raises = None

    @classmethod
    def last(cls) -> dict[str, Any]:
        assert cls.calls, "the orchestrator was never invoked"
        return cls.calls[-1]


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Neutral env + a scratch cwd, so no host key and no repo .env can leak in."""
    for name in KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith(FURROW_ENV_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC_KEY)
    # Settings reads .env relative to the cwd; run from a directory without one.
    monkeypatch.chdir(tmp_path)
    FakeOrchestrator.reset()
    yield


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def orchestrator(monkeypatch: pytest.MonkeyPatch) -> type[FakeOrchestrator]:
    monkeypatch.setattr(cli, "Orchestrator", FakeOrchestrator)
    return FakeOrchestrator


@pytest.fixture
def web_run(monkeypatch: pytest.MonkeyPatch) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    """Replace the web server entry point; returns the list of recorded calls."""
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(furrow.web.server, "run", lambda *a, **kw: calls.append((a, kw)))
    return calls


def invoke(runner: CliRunner, args: list[str], **kwargs: Any) -> Result:
    return runner.invoke(main, args, **kwargs)


# --------------------------------------------------------------------- start


def test_start_passes_the_goal_to_the_orchestrator(runner, orchestrator):
    result = invoke(runner, ["start", "Add JWT auth to the API"])

    assert result.exit_code == 0, result.output
    assert orchestrator.last()["goal"] == "Add JWT auth to the API"


def test_start_prompts_when_the_goal_is_missing(runner, orchestrator):
    result = invoke(runner, ["start"], input="Ship a changelog\n")

    assert result.exit_code == 0, result.output
    assert "Enter your goal for Furrow" in result.output
    assert orchestrator.last()["goal"] == "Ship a changelog"


def test_start_prompts_when_the_goal_is_blank(runner, orchestrator):
    result = invoke(runner, ["start", "   "], input="Refactor the parser\n")

    assert result.exit_code == 0, result.output
    assert orchestrator.last()["goal"] == "Refactor the parser"


def test_start_without_a_goal_and_no_input_fails_cleanly(runner, orchestrator):
    # Empty input ends the prompt (Ctrl-D); that must not look like success.
    result = invoke(runner, ["start"], input="")

    assert result.exit_code != 0
    assert orchestrator.calls == []


# ------------------------------------------------------------ settings plumbing


def test_flags_reach_the_settings_passed_to_the_orchestrator(
    runner, orchestrator, monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI_KEY)
    workspace = tmp_path / "repo"
    result = invoke(
        runner,
        [
            "start",
            "goal",
            "--provider",
            "openai",
            "--model",
            "gpt-fake",
            "--planner-model",
            "planner-fake",
            "--worker-model",
            "worker-fake",
            "--tester-model",
            "tester-fake",
            "--max-cycles",
            "4",
            "--max-parallel-tasks",
            "9",
            "--workspace",
            str(workspace),
            "--log-level",
            "debug",
        ],
    )

    assert result.exit_code == 0, result.output
    settings = orchestrator.last()["settings"]
    assert settings.provider is Provider.OPENAI
    assert settings.model == "gpt-fake"
    assert settings.planner_model == "planner-fake"
    assert settings.worker_model == "worker-fake"
    assert settings.tester_model == "tester-fake"
    assert settings.max_cycles == 4
    assert settings.max_parallel_tasks == 9
    assert settings.workspace == workspace
    assert settings.log_level == "DEBUG"


def test_each_flag_alone_reaches_the_settings(runner, orchestrator, tmp_path):
    cases = [
        (["--model", "solo-model"], "model", "solo-model"),
        (["--provider", "ollama"], "provider", Provider.OLLAMA),
        (["--max-cycles", "2"], "max_cycles", 2),
        (["--log-level", "warning"], "log_level", "WARNING"),
        (["--max-parallel-tasks", "3"], "max_parallel_tasks", 3),
    ]
    for flags, field, expected in cases:
        assert invoke(runner, ["start", "goal", *flags]).exit_code == 0
        assert getattr(orchestrator.last()["settings"], field) == expected


def test_env_and_dotenv_still_apply_when_no_flag_is_given(
    runner, orchestrator, monkeypatch, tmp_path
):
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "7")
    (tmp_path / ".env").write_text("FURROW_LOG_LEVEL=WARNING\nFURROW_MODEL=env-model\n")
    monkeypatch.chdir(tmp_path)

    assert invoke(runner, ["start", "goal"]).exit_code == 0

    settings = orchestrator.last()["settings"]
    assert settings.max_parallel_tasks == 7
    assert settings.log_level == "WARNING"
    assert settings.model == "env-model"


def test_max_cycles_zero_is_passed_through_as_unlimited(runner, orchestrator):
    result = invoke(runner, ["start", "goal", "--max-cycles", "0"])

    assert result.exit_code == 0, result.output
    assert orchestrator.last()["settings"].max_cycles == 0


def test_max_parallel_tasks_zero_is_passed_through(runner, orchestrator):
    assert invoke(runner, ["start", "goal", "--max-parallel-tasks", "0"]).exit_code == 0
    assert orchestrator.last()["settings"].max_parallel_tasks == 0


def test_negative_budgets_are_rejected(runner, orchestrator):
    result = invoke(runner, ["start", "goal", "--max-cycles", "-1"])

    assert result.exit_code != 0
    assert "--max-cycles" in result.output
    assert orchestrator.calls == []


def test_the_global_settings_singleton_is_never_mutated(runner, orchestrator, monkeypatch):
    singleton = furrow.config.settings
    before = singleton.model_dump()
    monkeypatch.setenv("FURROW_MODEL", "env-model")

    assert invoke(runner, ["start", "goal", "--model", "cli-model"]).exit_code == 0
    assert invoke(runner, ["start", "goal"]).exit_code == 0

    assert orchestrator.calls[0]["settings"].model == "cli-model"  # flag wins
    assert orchestrator.calls[1]["settings"].model == "env-model"  # then the environment
    assert furrow.config.settings is singleton
    assert singleton.model_dump() == before  # untouched by either


def test_settings_are_not_shared_between_invocations(runner, orchestrator):
    assert invoke(runner, ["start", "first", "--model", "one", "--max-cycles", "2"]).exit_code == 0
    assert invoke(runner, ["start", "second", "--max-cycles", "3"]).exit_code == 0

    first, second = orchestrator.calls
    assert first["settings"] is not second["settings"]
    assert first["settings"].model == "one"
    assert second["settings"].model == Settings().model
    assert second["settings"].max_cycles == 3


def test_build_settings_keeps_env_for_flags_that_were_not_given(monkeypatch):
    monkeypatch.setenv("FURROW_MODEL", "from-env")
    monkeypatch.setenv("FURROW_MAX_CYCLES", "5")

    settings = build_settings(model="from-flag", max_cycles=None)

    assert settings.model == "from-flag"
    assert settings.max_cycles == 5


def test_the_llm_client_receives_the_same_settings(runner, orchestrator):
    assert invoke(runner, ["start", "goal", "--provider", "ollama", "--model", "llama3"]).exit_code == 0

    call = orchestrator.last()
    assert isinstance(call["client"], cli.LLMClient)
    assert call["client"].settings is call["settings"]


# ---------------------------------------------------------------- failure modes


def test_invalid_provider_exits_nonzero_without_a_traceback(runner, orchestrator):
    result = invoke(runner, ["start", "goal", "--provider", "bedrock"])

    assert result.exit_code != 0
    assert "bedrock" in result.output
    assert "anthropic" in result.output  # lists the supported providers
    assert "Traceback" not in result.output
    assert orchestrator.calls == []


def test_unsupported_provider_in_the_environment_is_reported_cleanly(runner, orchestrator, monkeypatch):
    monkeypatch.setenv("FURROW_PROVIDER", "bedrock")
    result = invoke(runner, ["start", "goal"])

    assert result.exit_code != 0
    assert "invalid configuration" in result.output
    assert "provider" in result.output
    assert "Traceback" not in result.output
    assert orchestrator.calls == []


@pytest.mark.parametrize(
    ("provider", "key_env", "settings_field"),
    [
        ("anthropic", "ANTHROPIC_API_KEY", "anthropic_api_key"),
        ("openai", "OPENAI_API_KEY", "openai_api_key"),
    ],
)
def test_missing_api_key_is_actionable(
    runner, orchestrator, monkeypatch, provider, key_env, settings_field
):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("FURROW_ANTHROPIC_API_KEY", raising=False)

    result = invoke(runner, ["start", "goal", "--provider", provider])

    assert result.exit_code != 0
    assert "no API key found" in result.output
    assert key_env in result.output
    assert "FURROW_" in result.output
    assert "Traceback" not in result.output
    assert orchestrator.calls == []  # never gets as far as a request


def test_a_key_from_the_environment_lets_the_run_start(runner, orchestrator, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-a-real-key")

    assert invoke(runner, ["start", "goal"]).exit_code == 0
    assert orchestrator.last()["settings"].provider is Provider.ANTHROPIC


def test_a_key_in_dotenv_lets_the_run_start(runner, orchestrator, tmp_path):
    (tmp_path / ".env").write_text("FURROW_OPENAI_API_KEY=sk-not-a-real-key\n")

    assert invoke(runner, ["start", "goal", "--provider", "openai"]).exit_code == 0


def test_the_missing_key_error_never_prints_a_secret(runner, orchestrator, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-appear-anywhere")

    result = invoke(runner, ["start", "goal", "--provider", "openai"])

    assert result.exit_code != 0
    assert "sk-should-not-appear-anywhere" not in result.output


def test_ollama_needs_no_api_key(runner, orchestrator):
    assert invoke(runner, ["start", "goal", "--provider", "ollama"]).exit_code == 0


def test_keyboard_interrupt_exits_zero_with_a_message(runner, orchestrator):
    orchestrator.raises = KeyboardInterrupt()

    result = invoke(runner, ["start", "goal"])

    assert result.exit_code == 0
    assert "stopped by user" in result.output
    assert "Traceback" not in result.output


def test_start_help_documents_every_option(runner, orchestrator):
    result = invoke(runner, ["start", "--help"])

    assert result.exit_code == 0
    for option in (
        "--provider",
        "--model",
        "--planner-model",
        "--worker-model",
        "--tester-model",
        "--max-cycles",
        "--max-parallel-tasks",
        "--workspace",
        "--log-level",
    ):
        assert option in result.output


# ----------------------------------------------------------------------- web


def test_web_forwards_host_and_port(runner, web_run):
    result = invoke(runner, ["web", "--host", "0.0.0.0", "--port", "9001"])

    assert result.exit_code == 0, result.output
    assert web_run == [((), {"host": "0.0.0.0", "port": 9001})]


def test_web_defaults_to_loopback(runner, web_run):
    assert invoke(runner, ["web"]).exit_code == 0

    assert web_run == [((), {"host": WEB_HOST, "port": WEB_PORT})]
    assert web_run[0][1]["host"] == "127.0.0.1"
    assert web_run[0][1]["port"] == 8000


def test_web_keyboard_interrupt_exits_zero(runner, monkeypatch):
    def interrupt(*_: Any, **__: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(furrow.web.server, "run", interrupt)

    result = invoke(runner, ["web"])

    assert result.exit_code == 0
    assert "stopped by user" in result.output


# -------------------------------------------------------------------- version


def test_version_prints_the_package_version(runner):
    result = invoke(runner, ["--version"])

    assert result.exit_code == 0
    assert "furrow" in result.output
    assert get_version() in result.output


def test_version_looks_like_a_version(runner):
    assert re.search(r"furrow \d+\.\d+", invoke(runner, ["--version"]).output)


def test_get_version_falls_back_when_the_distribution_is_missing(monkeypatch):
    from importlib import metadata

    def missing(_name: str) -> str:
        raise metadata.PackageNotFoundError("furrow")

    monkeypatch.setattr(cli.metadata, "version", missing)

    version = get_version()

    assert version
    assert version[0].isdigit()


# ------------------------------------------------------------- import hygiene


def test_importing_the_cli_starts_nothing():
    # A subprocess, so the check cannot pass because another test already imported it.
    code = (
        "import uvicorn\n"
        "started = []\n"
        "uvicorn.run = lambda *a, **k: started.append(a)\n"
        "uvicorn.Server = lambda *a, **k: started.append(a)\n"
        "import furrow.cli.main\n"
        "assert not started, 'importing the CLI started a server'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr


def test_importing_the_cli_does_not_pull_in_the_web_server():
    code = "import sys, furrow.cli.main; assert 'furrow.web.server' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr


def test_group_help_lists_both_commands(runner):
    result = invoke(runner, ["--help"])

    assert result.exit_code == 0
    assert "start" in result.output
    assert "web" in result.output
    assert "--version" in result.output