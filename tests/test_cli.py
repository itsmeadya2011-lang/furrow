"""Tests for :mod:`furrow.cli.main`.

Two regressions are pinned here. ``--model`` used to be a total no-op: it set
``settings.model`` while every call site passed ``planner_model``/``worker_model``/
``tester_model``, so the flag changed nothing. And the command used to exit 0 no matter how the
run ended, which makes a script wrapping ``furrow start`` unable to trust its status.

Most tests drive the command through :class:`click.testing.CliRunner` with the orchestrator
replaced by a stub that reports a crafted :class:`LoopResult`, so the loop never runs and no
provider is ever contacted. The ``--model`` tests instead install loop doubles that record the
model each role was built with, because that flag is only observable from inside the loop.

``pytest`` collects classes named ``Test*``; ``TestResult`` is imported aliased for exactly that
reason (``conftest.py`` additionally opts it out for the whole suite).

Note on ``reset_settings``: ``get_settings`` caches the process-wide settings, so a second
invocation inside one test would inherit the first one's overrides. Every :func:`invoke` resets
the cache first, which is what a real shell invocation gets from a fresh process.
"""

from __future__ import annotations

import importlib
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import click
import pytest
from click.testing import CliRunner, Result

from furrow.config import Plan, Provider, Settings, TaskModel, reset_settings
from furrow.config import TestResult as _TestResult
from furrow.core import orchestrator as orchestrator_module
from furrow.core.orchestrator import LoopResult
from furrow.llm import LLMClient, LLMError
from furrow.web import server
from tests.conftest import build_settings, make_plan, passing

# ``furrow/cli/__init__.py`` re-exports the click group as ``furrow.cli.main``, which shadows the
# submodule of the same name, so the module object has to be fetched explicitly.
cli = importlib.import_module("furrow.cli.main")

REPO_ROOT = Path(__file__).resolve().parent.parent
runner = CliRunner()

#: Every ``start`` flag, verified against the parser below and against ``furrow start --help``.
#: ``--help`` is click's own option and therefore not part of the declared parameters.
DOCUMENTED_START_FLAGS = {
    "--provider",
    "--model",
    "--planner-model",
    "--worker-model",
    "--tester-model",
    "--max-cycles",
    "--parallel",
    "--workspace",
    "--log-level",
    "--debug",
}

#: The same for ``web``.
DOCUMENTED_WEB_FLAGS = {"--host", "--port", "--log-level"}


# --------------------------------------------------------------------------------------
# Harness: record what the command built, dictate how the run ends
# --------------------------------------------------------------------------------------


def invoke(*args: str, text: str | None = None, standalone: bool = True) -> Result:
    """Invoke the CLI group the way a shell would, with a fresh settings cache.

    ``standalone=False`` lets click let an exception reach the caller instead of turning it
    into a status code, which is how the ``click.UsageError`` itself can be inspected.
    """
    reset_settings()
    return runner.invoke(cli.main, [*args], input=text, standalone_mode=standalone)


class RecordedLLMClient(LLMClient):
    """The real client, so settings plumbing stays honest, with ``complete`` disabled.

    ``complete`` is the only method that could reach a provider, and every test in this module
    stubs the loop, so a call here would be a bug rather than a slow test.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings=settings)
        self.prompts: list[str] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.prompts.append(prompt)
        raise AssertionError("no CLI test may reach an LLM")


@dataclass
class Harness:
    """What ``furrow start`` built, plus the outcome the stubbed run should report."""

    goals: list[str] = field(default_factory=list)
    orchestrators: list[Any] = field(default_factory=list)
    clients: list[RecordedLLMClient] = field(default_factory=list)
    outcome: LoopResult | BaseException = field(
        default_factory=lambda: LoopResult(
            goal="a goal", cycles=1, stopped_reason="goal-complete", passed=True
        )
    )

    @property
    def settings(self) -> Settings:
        """The settings this run was built from."""
        assert self.clients, "the command never built an LLM client"
        settings = self.clients[-1].settings
        assert settings is not None
        return settings

    @property
    def prompts(self) -> list[str]:
        """Every prompt this run would have sent to a provider."""
        return [prompt for client in self.clients for prompt in client.prompts]


@pytest.fixture(autouse=True)
def cli_environment(monkeypatch: pytest.MonkeyPatch, tmp_workspace: Path) -> None:
    """Fix the terminal width (rich wraps output) and keep the workspace inside ``tmp_path``."""
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("FURROW_WORKSPACE", str(tmp_workspace))


@pytest.fixture(autouse=True)
def isolated_root_logging() -> Iterator[None]:
    """Undo ``logging.basicConfig``: ``_configure_logging`` is called by every ``start``."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    try:
        yield
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Iterator[Harness]:
    """Replace the run with a stub, and give the command the credentials it preflights for."""
    state = Harness()

    class StubOrchestrator:
        def __init__(self, goal: str, client: Any = None, on_event: Any = None) -> None:
            state.goals.append(goal)
            state.orchestrators.append(self)
            self.goal = goal
            self.client = client
            self.on_event = on_event

        async def run(self) -> LoopResult:
            if isinstance(state.outcome, BaseException):
                raise state.outcome
            # The report must show the goal the user typed, not the stub's placeholder.
            return replace(state.outcome, goal=self.goal)

    def build_client(settings: Settings | None = None) -> LLMClient:
        client = RecordedLLMClient(settings)
        state.clients.append(client)
        return client

    for provider in ("ANTHROPIC", "OPENAI"):
        monkeypatch.setenv(f"FURROW_{provider}_API_KEY", "cli-test-key")
    monkeypatch.setattr(cli, "Orchestrator", StubOrchestrator)
    monkeypatch.setattr(cli, "LLMClient", build_client)
    yield state


@dataclass
class RecordedModels:
    """The model each role's agent was constructed with."""

    planner: list[str | None] = field(default_factory=list)
    worker: list[str | None] = field(default_factory=list)
    tester: list[str | None] = field(default_factory=list)

    @property
    def roles(self) -> tuple[str | None, str | None, str | None]:
        assert self.planner and self.worker and self.tester, "the loop never built every role"
        return (self.planner[0], self.worker[0], self.tester[0])


@pytest.fixture
def role_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_workspace: Path,
) -> Iterator[RecordedModels]:
    """Install loop doubles that record the model their role is configured with.

    The real agents resolve their model with ``getattr(client.settings, "<role>_model", None)``,
    so reading the same attribute here asserts the plumbing the CLI is responsible for. The
    doubles still let the loop run to completion, which is why no orchestrator is stubbed.
    """
    seen = RecordedModels()

    class Planner:
        def __init__(self, client: Any = None, settings: Any = None) -> None:
            seen.planner.append(getattr(client.settings, "planner_model", None))

        async def plan(self, goal: str) -> Plan:
            return make_plan("do the work")

    class Worker:
        def __init__(self, task: Any, client: Any = None, settings: Any = None) -> None:
            seen.worker.append(getattr(client.settings, "worker_model", None))

        async def run(self) -> str:
            return "done"

    class Tester:
        def __init__(self, client: Any = None, settings: Any = None) -> None:
            seen.tester.append(getattr(client.settings, "tester_model", None))

        async def run(self, goal: str, tasks: list[TaskModel]) -> _TestResult:
            return passing()

    monkeypatch.setattr(orchestrator_module, "PlannerAgent", Planner)
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", Worker)
    monkeypatch.setattr(orchestrator_module, "TesterAgent", Tester)
    monkeypatch.setenv("FURROW_ANTHROPIC_API_KEY", "cli-test-key")
    monkeypatch.setenv("FURROW_WORKSPACE", str(tmp_workspace))
    yield seen


@pytest.fixture
def web_runner(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture the keyword arguments ``furrow web`` hands to the server runner."""
    calls: list[dict[str, Any]] = []

    def fake_run(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(server, "run", fake_run)
    return calls


def declared_flags(command: click.Command) -> set[str]:
    """The option strings a command's parser declares (its positional argument is not one)."""
    return {
        option
        for param in command.params
        if isinstance(param, click.Option)
        for option in param.opts
    }


def console_script_target() -> str:
    """The target of the ``furrow`` console script declared in ``pyproject.toml``."""
    sections = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    scripts = sections.split("[project.scripts]", 1)[1].split("[", 1)[0]
    match = re.search(r'^\s*furrow\s*=\s*"([^"]+)"', scripts, re.MULTILINE)
    assert match, "pyproject.toml must declare a [project.scripts] entry point for furrow"
    return match.group(1)


# --------------------------------------------------------------------------------------
# Help and the entry point
# --------------------------------------------------------------------------------------


def test_main_is_a_click_group() -> None:
    assert isinstance(cli.main, click.Group)
    assert set(cli.main.commands) == {"start", "web"}


def test_the_group_help_lists_both_commands() -> None:
    result = invoke("--help")

    assert result.exit_code == 0
    assert "start" in result.output
    assert "web" in result.output


def test_start_declares_every_documented_flag() -> None:
    assert declared_flags(cli.start) >= DOCUMENTED_START_FLAGS


def test_web_declares_every_documented_flag() -> None:
    assert declared_flags(cli.web) >= DOCUMENTED_WEB_FLAGS


def test_start_help_lists_every_option_the_parser_declares() -> None:
    """Derive the expectation from the parser, so the help text cannot drift away from it."""
    result = invoke("start", "--help")

    assert result.exit_code == 0
    for option in declared_flags(cli.start):
        assert option in result.output
    assert "--help" in result.output


def test_web_help_documents_host_and_port() -> None:
    result = invoke("web", "--help")

    assert result.exit_code == 0
    for option in declared_flags(cli.web):
        assert option in result.output


def test_web_help_warns_that_a_non_loopback_host_is_unauthenticated() -> None:
    result = invoke("web", "--help")

    assert "unauthenticated" in result.output
    assert "127.0.0.1" in result.output


def test_the_console_script_target_resolves() -> None:
    target = console_script_target()

    module_name, _, attribute = target.partition(":")
    assert (module_name, attribute) == ("furrow.cli.main", "main")
    assert getattr(importlib.import_module(module_name), attribute) is cli.main


# --------------------------------------------------------------------------------------
# The goal
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("goal", ["", "   ", "\n\t "], ids=["empty", "whitespace", "newlines"])
def test_a_blank_goal_is_a_usage_error(harness: Harness, goal: str) -> None:
    result = invoke("start", goal)

    assert result.exit_code == 2
    assert "non-empty goal is required" in result.output
    assert harness.goals == [], "a blank goal must not start a run"
    assert harness.clients == []


def test_a_blank_goal_is_reported_as_a_click_usage_error(harness: Harness) -> None:
    result = invoke("start", "   ")

    assert "Usage:" in result.output
    assert "Error: A non-empty goal is required" in result.output
    assert click.UsageError.exit_code == 2, "click's usage errors are what exit 2 means"


def test_a_blank_goal_raises_click_usage_error(harness: Harness) -> None:
    """With click's standalone handling off, the ``UsageError`` itself reaches the caller."""
    result = invoke("start", "   ", standalone=False)

    assert isinstance(result.exception, click.UsageError)
    assert str(result.exception).startswith("A non-empty goal is required")


def test_a_missing_goal_is_prompted_for(harness: Harness) -> None:
    result = invoke("start", text="ship it from the prompt\n")

    assert result.exit_code == 0
    assert harness.goals == ["ship it from the prompt"]


def test_a_blank_prompt_is_still_a_usage_error(harness: Harness) -> None:
    result = invoke("start", text="\n")

    assert result.exit_code == 2
    assert harness.goals == []


def test_a_prompted_goal_is_stripped(harness: Harness) -> None:
    result = invoke("start", text="  ship it  \n")

    assert result.exit_code == 0
    assert harness.goals == ["ship it"]


# --------------------------------------------------------------------------------------
# Credentials preflight
# --------------------------------------------------------------------------------------


def test_a_missing_api_key_exits_one_and_names_the_env_var(
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FURROW_ANTHROPIC_API_KEY", raising=False)

    result = invoke("start", "ship it")

    assert result.exit_code == 1
    assert "No API key for provider anthropic" in result.output
    assert "Set ANTHROPIC_API_KEY" in result.output
    assert "export ANTHROPIC_API_KEY=sk-..." in result.output


def test_a_missing_api_key_is_caught_before_any_llm_call(
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FURROW_ANTHROPIC_API_KEY", raising=False)

    invoke("start", "ship it")

    assert harness.prompts == [], "the preflight must run before the loop starts"
    assert harness.clients == [], "no client may be built without credentials"
    assert harness.goals == []


def test_a_missing_openai_key_names_the_openai_env_var(
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FURROW_OPENAI_API_KEY", raising=False)

    result = invoke("start", "ship it", "--provider", "openai")

    assert result.exit_code == 1
    assert "Set OPENAI_API_KEY" in result.output


def test_a_provider_without_a_key_needs_no_credentials(
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FURROW_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("FURROW_OPENAI_API_KEY", raising=False)

    result = invoke("start", "ship it", "--provider", "ollama")

    assert result.exit_code == 0, "ollama is local and needs no key"
    assert harness.goals == ["ship it"]


def test_the_key_may_come_from_the_plain_environment_variable(
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FURROW_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-the-shell")

    result = invoke("start", "ship it")

    assert result.exit_code == 0
    assert harness.goals == ["ship it"]


# --------------------------------------------------------------------------------------
# Exit codes
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stopped_reason", "passed", "exit_code"),
    [
        ("goal-complete", True, 0),
        ("goal-complete", False, 1),
        ("max-cycles", False, 1),
        ("stalled", False, 1),
        ("no-tasks", False, 1),
        ("cancelled", False, 1),
    ],
    ids=["complete", "complete-but-failed", "max-cycles", "stalled", "no-tasks", "cancelled"],
)
def test_the_exit_code_follows_the_run_outcome(
    harness: Harness,
    stopped_reason: str,
    passed: bool,
    exit_code: int,
) -> None:
    """``0`` is reserved for a goal that was genuinely completed with passing tests."""
    harness.outcome = LoopResult(
        goal="g", cycles=2, stopped_reason=stopped_reason, passed=passed, tasks_completed=1
    )

    result = invoke("start", "ship it")

    assert result.exit_code == exit_code


def test_a_completed_run_reports_success(harness: Harness) -> None:
    harness.outcome = LoopResult(
        goal="g", cycles=3, stopped_reason="goal-complete", passed=True, tasks_completed=2
    )

    result = invoke("start", "ship it")

    assert "Goal complete." in result.output
    assert "stopped because  goal-complete" in result.output
    assert "tasks completed  2" in result.output


def test_an_unfinished_run_says_so_and_prints_its_failures(harness: Harness) -> None:
    harness.outcome = LoopResult(
        goal="g",
        cycles=3,
        stopped_reason="stalled",
        passed=False,
        tasks_failed=1,
        failures=["tests/test_x.py::test_y"],
    )

    result = invoke("start", "ship it")

    assert "Stopped without completing the goal: stalled." in result.output
    assert "tests/test_x.py::test_y" in result.output


def test_a_keyboard_interrupt_exits_130(harness: Harness) -> None:
    """130 is the conventional SIGINT status, so a shell sees an interrupt, not a failure."""
    harness.outcome = KeyboardInterrupt()

    result = invoke("start", "ship it")

    assert result.exit_code == 130
    assert "stopped by user" in result.output


def test_an_llm_error_is_reported_as_a_failed_request(harness: Harness) -> None:
    harness.outcome = LLMError("the provider returned 500 after 3 attempts")

    result = invoke("start", "ship it")

    assert result.exit_code == 1
    assert "LLM request failed" in result.output
    assert "the provider returned 500 after 3 attempts" in result.output


def test_a_plain_value_error_is_reported_as_a_configuration_error(harness: Harness) -> None:
    harness.outcome = ValueError("OPENAI_API_KEY is not set")

    result = invoke("start", "ship it")

    assert result.exit_code == 1
    assert "Configuration error" in result.output
    assert "OPENAI_API_KEY is not set" in result.output


def test_an_unexpected_error_hides_the_traceback_and_points_at_debug(harness: Harness) -> None:
    harness.outcome = RuntimeError("the scheduler fell over")

    result = invoke("start", "ship it")

    assert result.exit_code == 1
    assert "Furrow failed" in result.output
    assert "--debug" in result.output


def test_an_unexpected_error_never_dumps_a_traceback_to_a_plain_user(harness: Harness) -> None:
    harness.outcome = RuntimeError("the scheduler fell over")

    result = invoke("start", "ship it")

    assert "Traceback" not in result.output
    assert "orchestrator.py" not in result.output
    assert "test_cli.py" not in result.output


def test_debug_prints_the_traceback(harness: Harness) -> None:
    harness.outcome = RuntimeError("the scheduler fell over")

    result = invoke("start", "ship it", "--debug")

    assert result.exit_code == 1
    assert "Traceback" in result.output
    assert "RuntimeError: the scheduler fell over" in result.output


def test_the_api_key_hint_follows_an_llm_error(harness: Harness, monkeypatch: Any) -> None:
    monkeypatch.delenv("FURROW_ANTHROPIC_API_KEY", raising=False)
    harness.outcome = LLMError("no route to the provider")

    result = invoke("start", "ship it")

    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.output


# --------------------------------------------------------------------------------------
# Models: --model used to be a no-op
# --------------------------------------------------------------------------------------


def test_the_model_flag_reaches_every_role(role_models: RecordedModels) -> None:
    result = invoke("start", "ship it", "--model", "house-model")

    assert result.exit_code == 0
    assert role_models.roles == ("house-model", "house-model", "house-model")


def test_the_model_flag_also_sets_the_fallback_model(role_models: RecordedModels) -> None:
    """Every call site passes a role model, but ``Settings.model`` is the documented default."""
    result = invoke("start", "ship it", "--model", "house-model")

    assert result.exit_code == 0
    assert role_models.roles == ("house-model",) * 3


def test_an_explicit_planner_model_overrides_the_model_flag(role_models: RecordedModels) -> None:
    result = invoke("start", "ship it", "--model", "X", "--planner-model", "Y")

    assert result.exit_code == 0
    assert role_models.planner == ["Y"]
    assert role_models.worker == ["X"]
    assert role_models.tester == ["X"]


def test_an_explicit_worker_model_overrides_the_model_flag(role_models: RecordedModels) -> None:
    result = invoke("start", "ship it", "--model", "X", "--worker-model", "W")

    assert result.exit_code == 0
    assert role_models.planner == ["X"]
    assert role_models.worker == ["W"]
    assert role_models.tester == ["X"]


def test_an_explicit_tester_model_overrides_the_model_flag(role_models: RecordedModels) -> None:
    result = invoke("start", "ship it", "--model", "X", "--tester-model", "T")

    assert result.exit_code == 0
    assert role_models.planner == ["X"]
    assert role_models.worker == ["X"]
    assert role_models.tester == ["T"]


def test_role_flags_apply_without_a_model_flag(role_models: RecordedModels) -> None:
    result = invoke("start", "ship it", "--worker-model", "W", "--tester-model", "T")

    assert result.exit_code == 0
    assert role_models.planner == [build_settings().planner_model]
    assert role_models.worker == ["W"]
    assert role_models.tester == ["T"]


def test_without_a_model_flag_the_configured_role_models_are_used(
    role_models: RecordedModels,
) -> None:
    result = invoke("start", "ship it")
    defaults = build_settings()

    assert result.exit_code == 0
    assert role_models.roles == (
        defaults.planner_model,
        defaults.worker_model,
        defaults.tester_model,
    )


def test_role_models_from_the_environment_survive_no_flags(
    role_models: RecordedModels,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FURROW_WORKER_MODEL", "env-worker")

    result = invoke("start", "ship it")

    assert result.exit_code == 0
    assert role_models.worker == ["env-worker"]


def test_a_model_containing_markup_is_passed_through_verbatim(harness: Harness) -> None:
    """The CLI never echoes the model, so verbatim is all that is required of it."""
    result = invoke("start", "ship it", "--model", "[bold]m")

    assert result.exit_code == 0
    assert harness.settings.model == "[bold]m"
    assert harness.settings.planner_model == "[bold]m"


# --------------------------------------------------------------------------------------
# The other flags reach the settings
# --------------------------------------------------------------------------------------


def test_the_provider_flag_reaches_the_settings(harness: Harness) -> None:
    result = invoke("start", "ship it", "--provider", "openai")

    assert result.exit_code == 0
    assert harness.settings.provider is Provider.OPENAI


def test_the_max_cycles_flag_reaches_the_settings(harness: Harness) -> None:
    result = invoke("start", "ship it", "--max-cycles", "4")

    assert result.exit_code == 0
    assert harness.settings.max_cycles == 4


def test_the_parallel_flag_reaches_max_parallel_tasks(harness: Harness) -> None:
    result = invoke("start", "ship it", "--parallel", "2")

    assert result.exit_code == 0
    assert harness.settings.max_parallel_tasks == 2


def test_the_workspace_flag_reaches_the_settings(harness: Harness, tmp_workspace: Path) -> None:
    result = invoke("start", "ship it", "--workspace", str(tmp_workspace))

    assert result.exit_code == 0
    assert harness.settings.workspace == tmp_workspace


def test_the_log_level_flag_reaches_the_settings(harness: Harness) -> None:
    result = invoke("start", "ship it", "--log-level", "DEBUG")

    assert result.exit_code == 0
    assert harness.settings.log_level == "DEBUG"


def test_settings_are_handed_to_the_client_rather_than_globally_mutated(
    harness: Harness,
) -> None:
    """The run must not depend on the cached singleton staying untouched afterwards."""
    invoke("start", "ship it", "--max-cycles", "6")

    reset_settings()
    assert build_settings().max_cycles == 0


def test_an_unusable_log_level_falls_back_to_info(
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``log_level`` is a free-form string in Settings, so a typo must not crash the run."""
    monkeypatch.setenv("FURROW_LOG_LEVEL", "chatty")

    result = invoke("start", "ship it")

    assert result.exit_code == 0
    assert "Unknown log level" in result.output


def test_the_run_never_reaches_a_provider(harness: Harness) -> None:
    invoke("start", "ship it")

    assert harness.prompts == []
    assert len(harness.orchestrators) == 1


# --------------------------------------------------------------------------------------
# Output is user text, not markup
# --------------------------------------------------------------------------------------


def test_a_goal_containing_markup_is_printed_literally(harness: Harness) -> None:
    goal = "add [bold]tags[/bold] to the [i]page[/i]"

    result = invoke("start", goal)

    assert result.exit_code == 0
    assert goal in result.output
    assert "\x1b[" not in result.output, "rich must not emit styling for a non-terminal stream"


def test_a_failure_line_containing_markup_is_printed_literally(harness: Harness) -> None:
    failure = "[bold]assert 1 == 2[/bold] in tests/test_x.py"
    harness.outcome = LoopResult(
        goal="g", cycles=1, stopped_reason="stalled", passed=False, failures=[failure]
    )

    result = invoke("start", "ship it")

    assert result.exit_code == 1
    assert failure in result.output
    assert "\x1b[" not in result.output


def test_the_start_banner_names_the_goal(harness: Harness) -> None:
    result = invoke("start", "ship the thing")

    assert "Furrow" in result.output
    assert "ship the thing" in result.output


# --------------------------------------------------------------------------------------
# furrow web
# --------------------------------------------------------------------------------------


def test_web_forwards_host_and_port_to_the_runner(web_runner: list[dict[str, Any]]) -> None:
    result = invoke("web", "--host", "0.0.0.0", "--port", "9999")

    assert result.exit_code == 0
    assert web_runner == [{"host": "0.0.0.0", "port": 9999, "log_level": None}]
    assert "http://0.0.0.0:9999" in result.output


def test_web_defaults_to_the_loopback_host(web_runner: list[dict[str, Any]]) -> None:
    result = invoke("web")

    assert result.exit_code == 0
    assert web_runner == [
        {"host": server.DEFAULT_HOST, "port": server.DEFAULT_PORT, "log_level": None}
    ]
    assert server.DEFAULT_HOST == "127.0.0.1"


def test_web_forwards_the_log_level(web_runner: list[dict[str, Any]]) -> None:
    result = invoke("web", "--log-level", "warning")

    assert result.exit_code == 0
    assert web_runner[0]["log_level"] == "warning"


def test_web_never_looks_for_credentials(
    web_runner: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serving the UI needs no key: there is nothing to call until a run is started."""
    monkeypatch.delenv("FURROW_ANTHROPIC_API_KEY", raising=False)

    result = invoke("web")

    assert result.exit_code == 0
    assert len(web_runner) == 1
