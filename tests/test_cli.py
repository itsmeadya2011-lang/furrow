"""Tests for the Click entry points: ``furrow/cli/main.py``.

Driven through :class:`click.testing.CliRunner` -- the contract is the process a user
sees: an exit code, a message, and whether a traceback escaped. The exit-code table is
``0`` success, ``1`` runtime failure (provider, planner, settings), ``2`` usage error
(bad flag or missing GOAL), ``130`` SIGINT.

Three fixtures do the setup work, all local to this file:

* ``api_key`` puts a dummy ``ANTHROPIC_API_KEY`` in the process environment **and** rebuilds
  the ``get_settings`` lru_cache. Passing ``env=`` to ``CliRunner.invoke`` is not enough:
  the cached singleton still holds the keyless configuration the process started with.
* ``fake_orchestrator`` replaces ``furrow.core.Orchestrator`` with a capturing fake. The CLI
  imports the name *inside* ``_run_loop``, so patching the attribute on the package is
  enough and is undone by ``monkeypatch``.
* ``colour`` pins the CLI's two module-level rich ``Console`` objects to 200 columns and
  forced ANSI colour, so the assertions about colourised task statuses do not depend on
  whether ``CliRunner`` happens to hand rich a TTY.

The regression tests worth naming: :func:`test_model_seeds_every_role_and_a_specific_flag_wins`
(``--model`` used to be a documented no-op because every agent passed an explicit model) and
:func:`test_the_settings_singleton_is_never_mutated` (the old code assigned onto the
process-wide pydantic singleton, leaking one run's flags into the next).
"""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from rich.color import ColorSystem

import furrow.agents as agents_pkg
import furrow.core as core_pkg
import furrow.web.server as web_server
from furrow.config import Settings, get_settings
from furrow.core import RunSummary
from furrow.llm import LLMError, ProviderAuthError
from furrow.models import Plan, TaskModel, TaskStatus

# `import furrow.cli.main as cli_mod` would grab the *Group* re-exported by
# `furrow/cli/__init__.py` (the package attribute shadows the submodule), so grab the real
# module explicitly.
cli_mod: Any = importlib.import_module("furrow.cli.main")
main: Any = cli_mod.main

#: Rich's "no colour" reset and the three SGR codes the status table must emit.
GREEN, RED, YELLOW, DIM = "\x1b[32m", "\x1b[31m", "\x1b[33m", "\x1b[2m"

#: The two module-level consoles ``_render_summary`` / ``_render_plan`` write through.
CLI_CONSOLES = (cli_mod.console, cli_mod.err_console)


def plain(text: str) -> str:
    """Strip SGR escapes.

    rich highlights numbers inside a message, so a raw substring match against prose fails
    whenever the phrase happens to contain a digit.
    """
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def no_traceback(result: Any) -> bool:
    """True when nothing resembling a Python traceback reached the captured output."""
    return "Traceback (most recent call last)" not in result.output


def invoke(*args: str, **kwargs: Any) -> Any:
    """Run the CLI with `args` in a fresh :class:`CliRunner`."""
    from click.testing import CliRunner

    return CliRunner().invoke(main, list(args), **kwargs)


def resolve_settings(**overrides: Any) -> Settings:
    """Call the CLI's own flag resolver with `overrides` (all others left as ``None``)."""
    keywords: dict[str, Any] = {
        "model": None,
        "planner_model": None,
        "worker_model": None,
        "tester_model": None,
        "provider": None,
        "workspace": None,
        "max_cycles": None,
        "max_parallel_tasks": None,
        "test_command": None,
        "test_timeout": None,
        "log_level": None,
    }
    keywords.update(overrides)
    return cli_mod._resolve_settings(**keywords)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def colour() -> Iterator[None]:
    """Pin the CLI's rich consoles to 200 columns with real ANSI colour.

    The consoles are built at *import* time, long before ``CliRunner`` replaces
    ``sys.stdout``, so they were auto-detected as non-terminals and emit no SGR at all --
    which would make any assertion about colourised status vacuous. ``_environ`` is what
    rich consults for ``COLUMNS`` and ``FORCE_COLOR`` and ``_color_system`` short-circuits
    terminal detection, so pinning both makes the rendering deterministic and then puts
    them back.
    """
    saved = [(console._environ, console._color_system) for console in CLI_CONSOLES]
    for console in CLI_CONSOLES:
        console._environ = {**console._environ, "COLUMNS": "200", "FORCE_COLOR": "1"}
        console._color_system = ColorSystem.TRUECOLOR
    yield
    for console, (environ, color_system) in zip(CLI_CONSOLES, saved, strict=True):
        console._environ = environ
        console._color_system = color_system


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A dummy Anthropic credential in both the environment and the settings cache.

    A real ``furrow`` invocation reads the environment before import, which is what
    rebuilding the ``lru_cache``d singleton reproduces. The rebuild on teardown keeps the
    credential out of any test that does not ask for one.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    get_settings(refresh=True)
    yield "sk-test-not-a-real-key"
    get_settings(refresh=True)


@pytest.fixture
def fake_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., dict[str, Any]]:
    """Factory installing a capturing stand-in for ``furrow.core.Orchestrator``.

    Returns a callable that takes the summary the fake should return (or the exception it
    should raise) and yields the capture dict the test asserts on::

        seen = fake_orchestrator(summary)
        result = invoke("start", "goal")
        assert seen["settings"].planner_model == "X"
    """

    def install(
        summary: RunSummary | None = None,
        *,
        raises: BaseException | None = None,
    ) -> dict[str, Any]:
        """Patch in the fake and return the dict its constructor/``run`` record into."""
        seen: dict[str, Any] = {}

        class CapturingOrchestrator:
            """Records how it was built; returns (or raises) exactly what the test wants."""

            def __init__(self, goal: str, client: Any = None, settings: Any = None, **kw: Any):
                seen["goal"] = goal
                seen["settings"] = settings
                seen["kwargs"] = kw

            async def run(self) -> RunSummary:
                if raises is not None:
                    raise raises
                return summary if summary is not None else make_summary()

        monkeypatch.setattr(core_pkg, "Orchestrator", CapturingOrchestrator)
        return seen

    return install


def make_summary(stopped_because: str = "max-cycles") -> RunSummary:
    """A summary covering all four task-status renderings."""
    return RunSummary(
        goal="ship it",
        cycles=3,
        completed=2,
        failed=1,
        skipped=1,
        stopped_because=stopped_because,
        tasks=[
            TaskModel(
                id="t1",
                description="completed thing",
                status=TaskStatus.COMPLETED,
                result="wrote a.py",
            ),
            TaskModel(
                id="t2",
                description="failed thing",
                status=TaskStatus.FAILED,
                result="ValueError: nope",
            ),
            TaskModel(id="t3", description="skipped thing", status=TaskStatus.SKIPPED),
            TaskModel(id="t4", description="pending thing"),
        ],
    )


# --------------------------------------------------------------------------- #
# 1. help renders for the group and both commands
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("args", "needle"),
    [
        (("--help",), "infinite parallel development loop"),
        (("--help",), "start"),
        (("start", "--help"), "--dry-run"),
        (("start", "--help"), "--max-parallel-tasks"),
        (("web", "--help"), "--host"),
    ],
)
def test_help_renders_without_a_traceback(args: tuple[str, ...], needle: str) -> None:
    """Help is the first thing a tab-completion probe runs; it must never import the SDKs."""
    result = invoke(*args)
    assert result.exit_code == 0
    assert no_traceback(result)
    assert needle in result.output


def test_the_group_advertises_exactly_start_and_web() -> None:
    """No module attribute may shadow the ``main`` group: `furrow.cli.main` is both."""
    from furrow.cli import main as group

    assert group is main
    assert set(group.commands) == {"start", "web"}
    assert cli_mod._package_version()


# --------------------------------------------------------------------------- #
# 2. usage errors -- exit code 2, message on stderr, no traceback
# --------------------------------------------------------------------------- #


def test_a_missing_goal_is_a_usage_error() -> None:
    """``furrow start`` with no GOAL prints usage on stderr and exits 2."""
    result = invoke("start")
    assert result.exit_code == 2
    assert "Usage" in result.stderr
    assert "Missing GOAL" in plain(result.stderr)
    assert no_traceback(result)


@pytest.mark.parametrize(
    ("args", "needle"),
    [
        (("start", "g", "--max-cycles", "-1"), "--max-cycles must be >= 0"),
        (("start", "g", "--max-parallel-tasks", "0"), "--max-parallel-tasks must be >= 1"),
        (("web", "--port", "99999"), "between 1 and 65535"),
        (("start", "g", "--workspace", "/does/not/exist"), "not an existing directory"),
    ],
)
def test_out_of_range_flags_are_usage_errors(args: tuple[str, ...], needle: str) -> None:
    """Each bad flag value exits 2 with a message that names the knob and its bound."""
    result = invoke(*args)
    assert result.exit_code == 2, plain(result.output)
    assert needle in plain(result.stderr)
    assert no_traceback(result)


def test_usage_errors_are_reported_before_the_credential_check(
    api_key: str,
) -> None:
    """``_resolve_settings`` runs first, so a bad flag never reaches the provider check."""
    result = invoke("start", "g", "--max-cycles", "-1")
    assert result.exit_code == 2
    assert "--max-cycles must be >= 0" in plain(result.stderr)


# --------------------------------------------------------------------------- #
# 3. a missing credential is a runtime error, not a usage error
# --------------------------------------------------------------------------- #


def test_no_api_key_exits_one_and_names_both_env_vars() -> None:
    """The message must tell the user exactly which variables to set."""
    result = invoke("start", "g")
    out = plain(result.output)
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in out
    assert "FURROW_ANTHROPIC_API_KEY" in out
    assert "No API key configured" in out
    assert no_traceback(result)


def test_a_dry_run_still_validates_flags(api_key: str) -> None:
    """``--dry-run`` is not a bypass for flag validation (checked before the key)."""
    assert invoke("start", "g", "--dry-run", "--max-cycles", "-1").exit_code == 2


def test_a_dry_run_still_needs_a_credential() -> None:
    """The credential check runs before the planner, so no key means no provider call."""
    result = invoke("start", "g", "--dry-run")
    assert result.exit_code == 1
    assert "No API key configured" in plain(result.output)


# --------------------------------------------------------------------------- #
# 4. --model precedence -- the flag that used to be a no-op
# --------------------------------------------------------------------------- #


def test_model_seeds_every_role_and_a_specific_flag_wins(
    api_key: str, fake_orchestrator: Callable[..., dict[str, Any]]
) -> None:
    """``--model X`` must reach the planner, the worker *and* the tester.

    ``--model`` was documented as overriding every role while being a no-op in practice:
    each agent passed an explicit role model, so the generic ``settings.model`` was never
    consulted. Asserting on the settings the *Orchestrator* received -- not on the flag
    parser -- is what makes this a real test of the contract.
    """
    seen = fake_orchestrator()
    result = invoke("start", "ship it", "--model", "X", "--worker-model", "Y")

    assert result.exit_code == 0, plain(result.output)
    run_settings: Settings = seen["settings"]
    assert run_settings.planner_model == "X"
    assert run_settings.worker_model == "Y"
    assert run_settings.tester_model == "X"
    assert run_settings.model == "X"
    assert run_settings.models() == {"planner": "X", "worker": "Y", "tester": "X"}


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (
            {"model": "X"},
            {"model": "X", "planner_model": "X", "worker_model": "X", "tester_model": "X"},
        ),
        (
            {"model": "X", "worker_model": "Y"},
            {"model": "X", "planner_model": "X", "worker_model": "Y", "tester_model": "X"},
        ),
        (
            {"model": "X", "planner_model": "P"},
            {"model": "X", "planner_model": "P", "worker_model": "X", "tester_model": "X"},
        ),
        (
            {"model": "X", "tester_model": "T"},
            {"model": "X", "planner_model": "X", "worker_model": "X", "tester_model": "T"},
        ),
        ({"planner_model": "P"}, {"planner_model": "P"}),
        ({"worker_model": "W"}, {"worker_model": "W"}),
    ],
)
def test_every_flag_combination_resolves_as_documented(
    overrides: dict[str, str], expected: dict[str, str]
) -> None:
    """The precedence matrix: ``--model`` seeds the roles, the specific flags then win.

    Any role field the case does not name is asserted to still hold the singleton's value,
    so a flag cannot quietly start resetting the other two.
    """
    base = get_settings()
    resolved = resolve_settings(**overrides)
    for field, value in expected.items():
        assert getattr(resolved, field) == value
    for field in ("model", "planner_model", "worker_model", "tester_model"):
        if field not in expected:
            assert getattr(resolved, field) == getattr(base, field), field


def test_the_settings_singleton_is_never_mutated(
    api_key: str, fake_orchestrator: Callable[..., dict[str, Any]]
) -> None:
    """The run gets a deep copy; the process-wide singleton is left exactly as it was.

    ``get_settings()`` is an ``lru_cache``d singleton that the LLM client, the planner and
    the orchestrator all hold by reference, so ``settings.model = x`` reconfigured every
    other component in the process -- and leaked this run's flags into the next one.
    """
    seen = fake_orchestrator()
    singleton = get_settings()
    before = (singleton.model, singleton.planner_model, singleton.worker_model)

    result = invoke("start", "ship it", "--model", "X")

    assert result.exit_code == 0, plain(result.output)
    assert get_settings() is singleton, "the cached singleton object was replaced"
    assert (singleton.model, singleton.planner_model, singleton.worker_model) == before
    run_settings: Settings = seen["settings"]
    assert run_settings is not singleton
    assert run_settings.model == "X"
    assert run_settings.workspace == singleton.workspace


def test_flags_the_user_did_not_pass_keep_their_environment_values() -> None:
    """Only what was passed is applied; everything else keeps its configured value."""
    resolved = resolve_settings(model="X")
    singleton = get_settings()
    assert resolved.provider == singleton.provider
    assert resolved.max_cycles == singleton.max_cycles
    assert resolved.test_command == singleton.test_command
    assert resolved.log_level == singleton.log_level
    assert resolved.workspace == singleton.workspace


def test_every_start_flag_reaches_the_settings(api_key: str) -> None:
    """A flag that parses but does not arrive is the same bug as a missing flag."""
    resolved = resolve_settings(
        provider="openai",
        test_command="pytest -q",
        test_timeout=42,
        log_level="DEBUG",
        max_cycles=7,
        max_parallel_tasks=9,
        planner_model="P",
        tester_model="T",
    )
    assert resolved.provider.value == "openai"
    assert resolved.test_command == "pytest -q"
    assert resolved.test_timeout == 42
    assert resolved.log_level == "DEBUG"
    assert resolved.max_cycles == 7
    assert resolved.max_parallel_tasks == 9
    assert resolved.planner_model == "P"
    assert resolved.tester_model == "T"


def test_workspace_is_resolved_to_an_absolute_existing_directory(tmp_path: Path) -> None:
    """A relative ``..`` segment must not defeat the existence check."""
    nested = tmp_path / "sub"
    nested.mkdir()
    resolved = resolve_settings(workspace=nested / ".." / "sub")
    assert resolved.workspace == nested.resolve()
    assert resolved.workspace_path().is_dir()


# --------------------------------------------------------------------------- #
# 5. --dry-run
# --------------------------------------------------------------------------- #


def test_dry_run_prints_the_plan_and_touches_nothing(
    api_key: str,
    tmp_workspace: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plan-only invocation writes no file and constructs no agent with side effects.

    ``tmp_path`` is used directly (not ``tmp_workspace``) so the marker file written here
    can live next to the workspace without the two fixtures colliding.
    """
    marker = tmp_path / "pre-existing.txt"
    marker.write_text("untouched")
    plan = Plan(
        rationale="Split the parser out, then wire it.",
        tasks=[
            TaskModel(id="t1", description="extract parser", files=["a.py", "b.py"]),
            TaskModel(id="t2", description="wire parser", files=["c.py"], dependencies=["t1"]),
        ],
    )
    calls: dict[str, int] = {"planner": 0, "tester": 0, "orchestrator": 0, "worker": 0}

    class FakePlanner:
        """Returns the scripted plan; counts its own construction."""

        def __init__(self, client: Any = None, settings: Any = None) -> None:
            self.settings = settings

        async def plan(self, goal: str) -> Plan:
            calls["planner"] += 1
            return plan

    class Forbidden:
        """Any construction of this agent fails the test."""

        def __init__(self, *a: Any, **k: Any) -> None:
            calls["worker"] += 1
            raise AssertionError("a --dry-run must not construct this agent")

    class FakeOrchestrator(Forbidden):
        """Like :class:`Forbidden`, counted under its own key."""

        def __init__(self, *a: Any, **k: Any) -> None:
            calls["orchestrator"] += 1
            raise AssertionError("a --dry-run must not construct the Orchestrator")

    monkeypatch.setattr(agents_pkg, "PlannerAgent", FakePlanner)
    monkeypatch.setattr(agents_pkg, "TesterAgent", Forbidden)
    monkeypatch.setattr(agents_pkg, "WorkerAgent", Forbidden)
    monkeypatch.setattr(core_pkg, "Orchestrator", FakeOrchestrator)

    before = sorted(p.name for p in tmp_workspace.iterdir())
    result = invoke("start", "ship it", "--dry-run", "--workspace", str(tmp_workspace))
    after = sorted(p.name for p in tmp_workspace.iterdir())

    assert result.exit_code == 0, plain(result.output)
    out = plain(result.output)
    assert calls == {"planner": 1, "tester": 0, "orchestrator": 0, "worker": 0}
    assert before == after == []
    assert "extract parser" in out
    assert "wire parser" in out
    assert "a.py, b.py" in out
    assert "t1" in out  # the dependency column
    assert "Rationale" in out
    assert "Split the parser out" in out
    assert "Run summary" not in out, "a dry run has no summary to render"
    assert marker.read_text() == "untouched"


def test_ollama_needs_no_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider with no ``env_var`` skips the credential check entirely."""
    plan = Plan(rationale="r", tasks=[TaskModel(id="t1", description="a real task")])

    class FakePlanner:
        """Returns a one-task plan."""

        def __init__(self, client: Any = None, settings: Any = None) -> None:
            self.settings = settings

        async def plan(self, goal: str) -> Plan:
            return plan

    monkeypatch.setattr(agents_pkg, "PlannerAgent", FakePlanner)
    result = invoke("start", "g", "--dry-run", "--provider", "ollama")
    assert result.exit_code == 0, plain(result.output)
    assert "a real task" in plain(result.output)


# --------------------------------------------------------------------------- #
# 6. rendering a finished run
# --------------------------------------------------------------------------- #


def test_a_full_run_renders_counts_stop_reason_and_coloured_statuses(
    api_key: str, fake_orchestrator: Callable[..., dict[str, Any]]
) -> None:
    """Exit 0, every count, the stop reason, and one SGR colour per task status."""
    seen = fake_orchestrator()
    result = invoke("start", "ship it", "--max-cycles", "3")

    assert result.exit_code == 0, plain(result.output)
    assert seen["goal"] == "ship it"
    assert seen["settings"].max_cycles == 3

    out = result.output
    flat = plain(out)
    assert "stopped_because: max-cycles" in flat
    for metric in ("cycles", "completed", "failed", "skipped"):
        assert metric in flat
    for label in ("completed thing", "failed thing", "skipped thing", "pending thing"):
        assert label in flat
    assert "wrote a.py" in flat
    # completed=green, failed=red, skipped=yellow, pending=dim.
    assert GREEN in out
    assert RED in out
    assert YELLOW in out
    assert DIM in out


def test_json_output_is_exactly_the_run_summary(
    api_key: str, fake_orchestrator: Callable[..., dict[str, Any]]
) -> None:
    """``--json`` puts a parseable summary on stdout and nothing else."""
    summary = make_summary("tests-passed")
    fake_orchestrator(summary)

    result = invoke("start", "ship it", "--json")

    assert result.exit_code == 0, plain(result.output)
    parsed = json.loads(result.stdout)
    assert parsed == summary.to_dict()
    assert "Run summary" not in result.stdout
    assert result.stderr == ""
    assert json.loads(json.dumps(parsed)) == parsed


# --------------------------------------------------------------------------- #
# 7. runtime failures -> exit 1, and SIGINT -> 130
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raises", "needle"),
    [
        (ProviderAuthError("Anthropic rejected the API key: 401"), "Anthropic rejected"),
        (LLMError("the provider is unreachable"), "unreachable"),
        (ValueError("planner returned output that is not a Plan: bad json"), "not a Plan"),
        (RuntimeError("socket exploded"), "FURROW_DEBUG"),
    ],
)
def test_runtime_failures_exit_one_with_a_message_and_no_traceback(
    api_key: str,
    fake_orchestrator: Callable[..., dict[str, Any]],
    raises: BaseException,
    needle: str,
) -> None:
    """Every real failure mode is one actionable line, never a stack trace."""
    fake_orchestrator(raises=raises)
    result = invoke("start", "ship it")

    assert result.exit_code == 1
    assert needle in plain(result.output)
    assert no_traceback(result)
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_an_llm_error_also_suggests_where_to_check_the_credential(
    api_key: str, fake_orchestrator: Callable[..., dict[str, Any]]
) -> None:
    """The LLM branch carries the retry/credential hint the other branches do not."""
    fake_orchestrator(raises=ProviderAuthError("Anthropic rejected the API key: 401"))
    result = invoke("start", "ship it")
    assert "Check the provider credential" in plain(result.output)


def test_a_keyboard_interrupt_exits_130(
    api_key: str, fake_orchestrator: Callable[..., dict[str, Any]]
) -> None:
    """SIGINT maps to the shell's conventional 130, not to a generic failure."""
    fake_orchestrator(raises=KeyboardInterrupt())
    result = invoke("start", "ship it")
    assert result.exit_code == 130
    assert "Interrupted" in plain(result.output)
    assert no_traceback(result)


def test_furrow_debug_reraises_for_bug_reports(
    api_key: str,
    fake_orchestrator: Callable[..., dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The escape hatch must actually escape, or "re-run with FURROW_DEBUG=1" is a lie."""
    fake_orchestrator(raises=RuntimeError("socket exploded"))
    monkeypatch.setenv("FURROW_DEBUG", "1")

    with pytest.raises(RuntimeError, match="socket exploded"):
        invoke("start", "ship it", catch_exceptions=False)


def test_a_value_error_from_ensure_api_key_is_not_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A settings problem is exit 1; only a bad flag is exit 2."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    get_settings(refresh=True)
    result = invoke("start", "ship it")
    assert result.exit_code == 1
    assert no_traceback(result)
    get_settings(refresh=True)


# --------------------------------------------------------------------------- #
# 8. furrow web
# --------------------------------------------------------------------------- #


def test_web_warns_only_when_publishing_to_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wildcard host is a security decision and is announced; loopback is silent."""
    bound: list[tuple[str, int]] = []
    monkeypatch.setattr(web_server, "run", lambda host, port: bound.append((host, port)))

    local = invoke("web")
    wildcard = invoke("web", "--host", "0.0.0.0")
    wildcard_port = invoke("web", "--host", "0.0.0.0", "--port", "9001")

    assert [local.exit_code, wildcard.exit_code, wildcard_port.exit_code] == [0, 0, 0]
    assert bound == [("127.0.0.1", 8000), ("0.0.0.0", 8000), ("0.0.0.0", 9001)]
    assert "unauthenticated" in plain(wildcard.output)
    assert "unauthenticated" not in plain(local.output)
    assert "unauthenticated" in plain(wildcard_port.output)


@pytest.mark.parametrize("host", ["", "*", "::", "[::]"])
def test_every_wildcard_host_form_is_warned_about(
    monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    """The set of "publishes to everything" spellings, not just ``0.0.0.0``."""
    monkeypatch.setattr(web_server, "run", lambda host, port: None)
    result = invoke("web", "--host", host)
    assert result.exit_code == 0, plain(result.output)
    assert "unauthenticated" in plain(result.output)


def test_web_forwards_host_and_port_to_the_server_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI must not swallow the bind address; it hands it straight to ``run()``."""
    seen: dict[str, Any] = {}

    def fake_run(*, host: str, port: int) -> None:
        seen["host"] = host
        seen["port"] = port

    monkeypatch.setattr(web_server, "run", fake_run)
    result = invoke("web", "--host", "0.0.0.0", "--port", "9100")

    assert result.exit_code == 0, plain(result.output)
    assert seen == {"host": "0.0.0.0", "port": 9100}


def test_the_web_server_is_never_imported_at_module_scope() -> None:
    """``furrow --help`` must not pay for FastAPI, uvicorn and the provider SDK chain.

    The CLI's stated import policy is that :mod:`furrow.web.server` and :mod:`furrow.llm`
    are function-local, which is what keeps ``furrow --help`` at ~0.2 s instead of ~2.1 s.
    A module-scope import would silently undo that, and nothing else in the suite would
    notice.
    """
    source = Path(cli_mod.__file__).read_text(encoding="utf-8")
    module_level = [
        line
        for line in source.splitlines()
        if line.startswith(("from furrow.web", "from furrow.llm", "from furrow.core"))
    ]
    assert module_level == [], module_level


def test_furrow_version_works_without_an_installed_distribution() -> None:
    """This checkout has no ``furrow`` dist-info, so ``--version`` must still resolve."""
    result = invoke("--version")
    assert result.exit_code == 0
    assert no_traceback(result)
    assert result.output.strip() == f"furrow, version {cli_mod._package_version()}"
