"""Behavioural tests for :mod:`furrow.cli.main`.

The CLI is the boundary where a user finds out what the rest of the program
does, so its promises are the promises *of the messages, not the code*: a
missing credential reads as ``error: ...`` on stderr plus exit code 2, an
unparseable plan reads as a message, and a resource is released on every exit
path. These tests drive the real click group through
:class:`click.testing.CliRunner` and assert on exit codes and stderr text
rather than on internal calls, so a rewording of the implementation cannot
silently break them.

Three things make this file unusual, and each is pinned by a test of its own:

* **The package attribute shadows the module.**
  ``furrow/cli/__init__.py`` re-exports the click group as ``main``, so
  ``import furrow.cli.main as m`` binds the *Group*, not the module, and
  ``monkeypatch.setattr("furrow.cli.main.Orchestrator", ...)`` writes an
  attribute onto the Group object while the module -- the thing ``_drive``
  actually reads -- stays untouched. Every patch below therefore goes through
  the module object fetched from :data:`sys.modules`, and
  :func:`test_module_object_patching_replaces_the_symbols` proves the patch is
  live so this cannot quietly decay into a no-op that still passes.

* **``--model`` used to be dead.** It set the one model field no call site
  reads, so ``furrow start "..." --model gpt-4o`` silently ran Anthropic model
  ids. The four resolved model fields are asserted explicitly for every
  combination, and the ``furrow.config.settings`` singleton is snapshotted
  around each run because the CLI must never mutate global state.

* **Resources are released on every path.** The recorded event log proves the
  client is closed after a successful run, after an
  :class:`~furrow.errors.LLMError`, and -- the case that was a real leak --
  after ``Orchestrator(...)`` *construction* raises, before ``orchestrator`` is
  even bound.

Everything here is offline and deterministic: no network, no API key, and no
ambient ``FURROW_*`` environment leaking in (a module-autouse fixture clears
it).
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import click
import pytest
from click.testing import CliRunner, Result

import furrow.cli
from furrow.cli.main import main
from furrow.config import PROVIDER_MODELS, Provider, Settings
from furrow.core.orchestrator import Orchestrator
from furrow.errors import LLMError
from furrow.llm import LLMClient

#: The real ``furrow.cli.main`` *module*. Reached through ``sys.modules`` on
#: purpose -- see the module docstring. Attribute access on the package spells
#: ``furrow.cli.main``, which is the click Group and a different object.
MOD = sys.modules["furrow.cli.main"]

#: Shape used to prove the CLI builds its own Settings instead of poking the
#: process-wide singleton.
MODEL_FIELDS = ("model", "planner_model", "worker_model", "tester_model")

#: A syntactically plausible key. Never sent anywhere: the client and the
#: orchestrator are stubbed, this only satisfies the credential pre-flight.
FAKE_API_KEY = "sk-ant-FAKEKEY0001"

#: A key long enough that the last four characters are worth masking, plus a
#: substring that must never be echoed.
SECRET_WITH_TAIL = "sk-ant-SECRETVALUE1234"

#: Exit code click uses for a usage error, and the code the CLI reuses for an
#: expected failure (bad configuration, missing credentials, an LLM error).
EXIT_FAILURE = 2

#: 128 + SIGINT: "terminated by Ctrl-C", not success.
EXIT_INTERRUPTED = 130

#: Exact wording of the no-goal error, as a single string so a partial rewrite
#: cannot pass.
GOAL_REQUIRED = 'error: a goal is required: furrow start "<goal>"'

SHELL_CONFIRM_REQUIRED = (
    "error: --allow-shell needs explicit confirmation: pass --yes or "
    "--dangerously-allow-shell"
)


# ── stubs ───────────────────────────────────────────────────────────────


class _Recorder:
    """Shared, ordered log of everything the stubs observed.

    One mechanism covers all the assertions that matter: the settings the CLI
    resolved (captured at construction time), the goal it parsed, and the exact
    order in which the client, the orchestrator and their ``aclose`` calls
    happened. Ordering is the point -- "the client was closed" is only
    interesting next to "and nothing else was closed in the wrong order".

    ``ctor_error`` and ``run_error`` are raised from the corresponding stub
    call, which is how the teardown paths are exercised without a real loop.
    """

    def __init__(self) -> None:
        self.events: list[str] = []
        self.settings: list[Settings] = []
        self.goals: list[str] = []
        self.ctor_error: BaseException | None = None
        self.run_error: BaseException | None = None
        self.orchestrator_stub: type | None = None
        self.client_stub: type | None = None

    @property
    def resolved(self) -> Settings:
        """The Settings from the most recent run. Fails loudly if there was none."""
        assert self.settings, "the CLI never constructed an LLMClient; nothing to inspect"
        return self.settings[-1]

    @property
    def models(self) -> dict[str, str]:
        """The four model fields as ``{field: value}``, for readable asserts."""
        settings = self.resolved
        return {field: getattr(settings, field) for field in MODEL_FIELDS}


def _install_stubs(
    monkeypatch: pytest.MonkeyPatch,
    recorder: _Recorder,
) -> _Recorder:
    """Replace ``LLMClient`` and ``Orchestrator`` with recording stubs.

    Both are built as closures over ``recorder`` so settings capture, goal
    capture and ``aclose`` ordering all land in one log. The returned recorder
    also carries ``.orchestrator_stub`` / ``.client_stub``, the exact objects
    installed, so a test can assert the patch really took.

    The patches target the *module object* (see the module docstring). Setting
    them through a dotted string would silently write onto the click Group
    instead -- :func:`test_string_target_patching_is_a_silent_no_op` pins why.
    """

    class StubClient:
        """Stand-in for :class:`furrow.llm.LLMClient`. Makes no network calls."""

        async def aclose(self) -> None:
            recorder.events.append("client.aclose")

    class StubOrchestrator:
        """Stand-in for :class:`~furrow.core.orchestrator.Orchestrator`."""

        def __init__(
            self,
            *,
            goal: str,
            client: Any = None,
            settings: Settings | None = None,
            **kwargs: Any,
        ) -> None:
            if recorder.ctor_error is not None:
                # Construction failed, so there is no object to close -- exactly
                # the state that used to leak the client.
                recorder.events.append("orchestrator-ctor")
                raise recorder.ctor_error
            recorder.events.append("orchestrator")
            recorder.goals.append(goal)
            recorder.settings.append(settings)

        def request_stop(self, reason: str = "") -> None:
            """Record a stop request from the SIGINT handler."""
            recorder.events.append(f"request_stop:{reason}")

        async def run(self) -> None:
            """Record the run, then raise ``recorder.run_error`` if asked."""
            recorder.events.append("run")
            if recorder.run_error is not None:
                raise recorder.run_error

        async def aclose(self) -> None:
            recorder.events.append("orchestrator.aclose")

    def StubLLMClient(settings: Settings | None = None, **kwargs: Any) -> StubClient:
        """Factory the CLI calls as ``LLMClient(settings=...)``."""
        recorder.events.append("client")
        recorder.settings.append(settings)
        return StubClient()

    recorder.orchestrator_stub = StubOrchestrator
    recorder.client_stub = StubLLMClient
    monkeypatch.setattr(MOD, "LLMClient", StubLLMClient)
    monkeypatch.setattr(MOD, "Orchestrator", StubOrchestrator)
    return recorder


def _record_settings_resolution(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Spy on ``_resolve_settings`` and return the list it records into.

    One dict per call, holding the overrides the CLI passed. The list doubles as
    the proof that resolution happened *at all*: ``start --allow-shell`` with no
    confirmation must abort before this point, which is what makes
    ``allow_shell`` provably never ``True``.
    """
    calls: list[dict[str, Any]] = []
    real = MOD._resolve_settings

    def _recording(**overrides: Any) -> Settings:
        calls.append(dict(overrides))
        return real(**overrides)

    monkeypatch.setattr(MOD, "_resolve_settings", _recording)
    return calls


def _config_value(result: Result, name: str) -> str:
    """Return the value ``config`` printed for settings field ``name``."""
    for line in result.output.splitlines():
        if line.startswith(name):
            return line.partition("  ")[2].strip()
    raise AssertionError(f"config printed no line for {name!r}:\n{result.output}")


# ── fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _hermetic_cli(agent_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear ``FURROW_*`` and both API keys, then run from an empty directory.

    ``agent_env`` guarantees no ambient env var leaks in, and
    ``monkeypatch.chdir`` guarantees pydantic-settings finds no stray ``.env``
    file -- ``_resolve_settings`` builds ``Settings()`` *without*
    ``_env_file=None``, so a ``.env`` in the working directory would be read.
    """
    monkeypatch.chdir(agent_env)


@pytest.fixture
def recorder() -> _Recorder:
    """An empty :class:`_Recorder` for a test to fill and then assert on."""
    return _Recorder()


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch, recorder: _Recorder) -> _Recorder:
    """Stub out ``LLMClient`` and ``Orchestrator`` around a shared recorder."""
    return _install_stubs(monkeypatch, recorder)


@pytest.fixture
def keyed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set an API key *after* ``agent_env`` cleared it, reaching the run path.

    Ordering matters: ``agent_env`` deletes both key variables, so this must be
    requested after it (the autouse fixture runs first, so by the time a test
    body runs, setting the variable sticks).
    """
    monkeypatch.setenv("FURROW_ANTHROPIC_API_KEY", FAKE_API_KEY)


@contextlib.contextmanager
def _global_settings_untouched() -> Iterator[Settings]:
    """Assert the ``furrow.config.settings`` singleton is not touched inside.

    Identity *and* ``model_dump()``: rebinding the name would satisfy a weaker
    check that only compared values.
    """
    from furrow.config import settings as global_settings

    before = global_settings.model_dump()
    identity = id(global_settings)
    yield global_settings
    assert id(global_settings) == identity, "the CLI rebound furrow.config.settings"
    assert global_settings.model_dump() == before, "the CLI mutated global settings"


# ── the shadowing trap ──────────────────────────────────────────────────


def test_package_attribute_is_the_click_group() -> None:
    """``furrow.cli.main`` the *attribute* is the Group, not the module."""
    assert isinstance(furrow.cli.main, click.Group)
    assert furrow.cli.main is main
    assert furrow.cli.main.__class__ is click.Group


def test_import_statement_binds_the_group_and_hides_the_module() -> None:
    """``import furrow.cli.main as m`` yields the Group: the reason for ``MOD``."""
    import furrow.cli.main as m

    assert isinstance(m, click.Group), "expected the re-exported click Group"
    assert m is furrow.cli.main is main
    assert m is not MOD
    assert MOD.__name__ == "furrow.cli.main"
    assert type(MOD) is type(sys), "MOD should be a plain module object"
    assert not isinstance(MOD, click.Group)


def test_module_object_patching_replaces_the_symbols(
    monkeypatch: pytest.MonkeyPatch,
    stubs: _Recorder,
) -> None:
    """The stub really replaced the names ``_drive`` reads.

    Without this, a regression back to dotted-string patching would leave every
    other test asserting against a *real* client and orchestrator -- or, worse,
    pass while doing nothing.
    """
    assert stubs.orchestrator_stub is not None
    assert stubs.client_stub is not None
    assert MOD.Orchestrator is stubs.orchestrator_stub
    assert MOD.Orchestrator is not Orchestrator
    assert MOD.LLMClient is stubs.client_stub
    assert MOD.LLMClient is not LLMClient
    # The real symbols are still what the package imports, i.e. we replaced the
    # module binding and not some other reference.
    assert furrow.cli.main is not MOD.Orchestrator


def test_string_target_patching_is_a_silent_no_op(
    monkeypatch: pytest.MonkeyPatch,
    stubs: _Recorder,
) -> None:
    """Why the tests patch ``MOD``: a dotted string lands on the Group.

    This is the trap made executable. ``monkeypatch.setattr("furrow.cli.main.X",
    ...)`` walks attributes, so it finds the click Group rather than the module
    and sets the attribute there. The module -- and therefore ``_drive`` -- never
    notices. Asserting this keeps the shortcuts visibly broken.
    """
    sentinel = object()
    monkeypatch.setattr("furrow.cli.main.Orchestrator", sentinel, raising=False)

    assert getattr(furrow.cli.main, "Orchestrator", None) is sentinel
    assert MOD.Orchestrator is stubs.orchestrator_stub, "the module must be untouched"
    assert MOD.Orchestrator is not sentinel


def test_module_patching_is_undone_by_monkeypatch(
    recorder: _Recorder,
) -> None:
    """The module object really is mutable, and undo restores the real symbol."""
    patch = pytest.MonkeyPatch()
    try:
        patch.setattr(MOD, "Orchestrator", recorder.orchestrator_stub)
        assert MOD.Orchestrator is recorder.orchestrator_stub
    finally:
        patch.undo()
    assert MOD.Orchestrator is Orchestrator


# ── top-level plumbing ──────────────────────────────────────────────────


def test_version_option() -> None:
    result = CliRunner().invoke(main, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == "furrow, version 0.1.0"
    assert "Traceback" not in result.output


def test_help_lists_every_command() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    for command in ("start", "web", "config"):
        assert command in result.output
    assert "Traceback" not in result.output


def test_module_fields_constant_covers_all_four_model_fields() -> None:
    """``--model`` fans out through exactly these, so the tuple is load-bearing."""
    assert MOD.MODEL_FIELDS == MODEL_FIELDS


# ── goal resolution ─────────────────────────────────────────────────────


@pytest.mark.parametrize("goal", ["", "   "])
def test_start_requires_a_goal(goal: str) -> None:
    result = CliRunner().invoke(main, ["start", goal])

    assert result.exit_code == EXIT_FAILURE
    assert GOAL_REQUIRED in result.stderr
    assert "Traceback" not in result.output


def test_start_prompts_and_fails_when_stdin_is_empty() -> None:
    """No goal argument and no stdin to prompt with is still an error, not a hang."""
    result = CliRunner().invoke(main, ["start"], input="")

    assert result.exit_code == EXIT_FAILURE
    assert GOAL_REQUIRED in result.stderr
    assert "Traceback" not in result.output


# ── missing credentials ─────────────────────────────────────────────────


def test_start_without_a_key_reports_the_credential_problem() -> None:
    """Both the advisory warning and the hard error, so the fix is obvious."""
    result = CliRunner().invoke(main, ["start", "some goal"])

    assert result.exit_code == EXIT_FAILURE
    stderr = result.stderr
    assert "warning: provider is anthropic but no API key configured" in stderr
    assert "error: provider 'anthropic' needs an API key but none is configured" in stderr
    assert "hint: set ANTHROPIC_API_KEY, or FURROW_ANTHROPIC_API_KEY" in stderr
    assert "Traceback" not in result.output
    # An unusable configuration must not reach the client or the loop.
    assert "client" not in stderr


# ── --allow-shell confirmation ───────────────────────────────────────────


def test_allow_shell_without_confirmation_aborts_before_any_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--allow-shell`` on its own grants nothing, and never even resolves settings.

    The recorder is asserted *empty*, which is the strong form of the claim: if
    the CLI had resolved settings first and failed later, the flag would have
    been carried into a ``Settings`` object at some point. Zero calls proves
    ``allow_shell`` never became ``True`` anywhere.
    """
    calls = _record_settings_resolution(monkeypatch)

    result = CliRunner().invoke(main, ["start", "goal", "--allow-shell"])

    assert result.exit_code == EXIT_FAILURE
    assert SHELL_CONFIRM_REQUIRED in result.stderr
    assert "hint:" in result.stderr
    assert calls == []
    assert "Traceback" not in result.output


@pytest.mark.parametrize("confirm", ["--yes", "--dangerously-allow-shell"])
def test_confirmed_allow_shell_reaches_settings(
    confirm: str,
    monkeypatch: pytest.MonkeyPatch,
    stubs: _Recorder,
    keyed: None,
) -> None:
    """The non-vacuous control for the test above: with a confirmation flag the
    recorder IS called and ``allow_shell`` really is ``True``."""
    calls = _record_settings_resolution(monkeypatch)

    result = CliRunner().invoke(main, ["start", "goal", "--allow-shell", confirm])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["allow_shell"] is True
    assert stubs.resolved.allow_shell is True
    # Enabling it says so out loud, and points at where the grant came from.
    assert "shell access is enabled via --allow-shell" in result.stderr


def test_allow_shell_without_the_flag_never_sets_the_override(
    monkeypatch: pytest.MonkeyPatch,
    stubs: _Recorder,
    keyed: None,
) -> None:
    """The default grants nothing: ``allow_shell`` is ``None``, not ``False``.

    ``None`` means "the user did not ask", and ``_resolve_settings`` drops it,
    so an ``FURROW_ALLOW_SHELL`` in the environment still applies. Writing
    ``False`` here would silently strip that.
    """
    calls = _record_settings_resolution(monkeypatch)

    result = CliRunner().invoke(main, ["start", "goal"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["allow_shell"] is None
    assert stubs.resolved.allow_shell is False
    assert "shell access is enabled" not in result.stderr


# ── config: secrets must stay secret ────────────────────────────────────


def test_config_masks_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_ANTHROPIC_API_KEY", SECRET_WITH_TAIL)

    result = CliRunner().invoke(main, ["config"])

    assert result.exit_code == 0
    assert _config_value(result, "anthropic_api_key") == "set (****1234)"
    assert _config_value(result, "openai_api_key") == "not set"
    assert "set (****1234)" in result.output
    assert "not set" in result.output
    # The one assertion that would fail loudly if masking were ever removed.
    assert SECRET_WITH_TAIL not in result.output
    assert "SECRETVALUE" not in result.output
    assert "Traceback" not in result.output


def test_config_reports_no_missing_credential_when_a_key_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FURROW_ANTHROPIC_API_KEY", SECRET_WITH_TAIL)

    result = CliRunner().invoke(main, ["config"])

    assert result.exit_code == 0
    assert "warning" not in result.stderr
    assert "needs an API key" not in result.stderr


def test_config_warns_but_succeeds_when_no_key_is_configured() -> None:
    """``config`` is a diagnostic: it reports the problem instead of refusing."""
    result = CliRunner().invoke(main, ["config"])

    assert result.exit_code == 0
    assert _config_value(result, "anthropic_api_key") == "not set"
    assert _config_value(result, "openai_api_key") == "not set"
    assert "provider 'anthropic' needs an API key but none is configured" in result.stderr
    assert "Traceback" not in result.output


def test_config_never_echoes_a_short_key_in_full(monkeypatch: pytest.MonkeyPatch) -> None:
    """The last four characters *are* the key when it is this short."""
    monkeypatch.setenv("FURROW_ANTHROPIC_API_KEY", "abcd")

    result = CliRunner().invoke(main, ["config"])

    assert result.exit_code == 0
    assert _config_value(result, "anthropic_api_key") == "set"
    assert "abcd" not in result.output


def test_config_model_flag_fans_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """``config`` shares ``_model_overrides`` with ``start``, so the fix applies."""
    result = CliRunner().invoke(main, ["config", "--model", "flag-model"])

    assert result.exit_code == 0
    assert _config_value(result, "model") == "flag-model"
    assert _config_value(result, "planner_model") == "flag-model"
    assert _config_value(result, "worker_model") == "flag-model"
    assert _config_value(result, "tester_model") == "flag-model"


# ── the run path ────────────────────────────────────────────────────────


def test_success_path_runs_then_releases_everything(
    stubs: _Recorder,
    keyed: None,
) -> None:
    """Client, orchestrator, run, then both closes -- in that order."""
    with _global_settings_untouched():
        result = CliRunner().invoke(main, ["start", "ship it"])

    assert result.exit_code == 0, result.output
    assert stubs.events == [
        "client",
        "orchestrator",
        "run",
        "orchestrator.aclose",
        "client.aclose",
    ]
    assert "Traceback" not in result.output


def test_run_receives_the_parsed_goal(stubs: _Recorder, keyed: None) -> None:
    result = CliRunner().invoke(main, ["start", "  make it fast  "])

    assert result.exit_code == 0, result.output
    assert stubs.goals == ["make it fast"], "the goal must be stripped before use"


def test_settings_come_from_the_flags(stubs: _Recorder, keyed: None) -> None:
    result = CliRunner().invoke(
        main,
        [
            "start",
            "goal",
            "--max-cycles",
            "3",
            "--max-parallel-tasks",
            "2",
            "--worker-iterations",
            "5",
            "--test-command",
            "pytest -q",
        ],
    )

    assert result.exit_code == 0, result.output
    settings = stubs.resolved
    assert settings.max_cycles == 3
    assert settings.max_parallel_tasks == 2
    assert settings.worker_max_iterations == 5
    assert settings.test_command == "pytest -q"
    assert settings.allow_shell is False


def test_llm_error_from_run_is_reported_and_still_releases(
    stubs: _Recorder,
    keyed: None,
) -> None:
    """A provider failure is a message, and it must not skip the teardown."""
    stubs.run_error = LLMError("provider rejected the request")

    result = CliRunner().invoke(main, ["start", "goal"])

    assert result.exit_code == EXIT_FAILURE
    assert "error: provider rejected the request" in result.stderr
    assert stubs.events == [
        "client",
        "orchestrator",
        "run",
        "orchestrator.aclose",
        "client.aclose",
    ]
    assert "Traceback" not in result.output


def test_llm_error_hint_is_printed(stubs: _Recorder, keyed: None) -> None:
    stubs.run_error = LLMError("plan was unparseable", hint="re-run with --model")

    result = CliRunner().invoke(main, ["start", "goal"])

    assert result.exit_code == EXIT_FAILURE
    assert "error: plan was unparseable" in result.stderr
    assert "hint: re-run with --model" in result.stderr
    assert "Traceback" not in result.output


def test_unexpected_error_propagates(stubs: _Recorder, keyed: None) -> None:
    """A non-Furrow exception is NOT swallowed: it is a bug, and must surface."""
    stubs.run_error = RuntimeError("boom")

    result = CliRunner().invoke(main, ["start", "goal"])

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert result.exception.args == ("boom",)
    # Click parks the exception for the caller; the CLI converted nothing here.
    assert result.exc_info is not None
    assert result.exc_info[0] is RuntimeError
    # The teardown still ran on the way out.
    assert stubs.events == [
        "client",
        "orchestrator",
        "run",
        "orchestrator.aclose",
        "client.aclose",
    ]


def test_keyboard_interrupt_exits_130(stubs: _Recorder, keyed: None) -> None:
    """Ctrl-C is 130, not 0: a half-finished run did not succeed."""
    stubs.run_error = KeyboardInterrupt()

    result = CliRunner().invoke(main, ["start", "goal"])

    assert result.exit_code == EXIT_INTERRUPTED
    assert "warning: interrupted; stopping" in result.stderr
    assert "client.aclose" in stubs.events
    assert "orchestrator.aclose" in stubs.events
    assert "Traceback" not in result.output


def test_client_is_closed_when_orchestrator_construction_fails(
    stubs: _Recorder,
    keyed: None,
) -> None:
    """The leak that was found and fixed.

    ``_drive`` builds the client and *then* constructs the orchestrator. If the
    construction raises, ``orchestrator`` was never bound -- but the client has
    already opened an httpx connection pool, so it must still be closed.
    """
    stubs.ctor_error = LLMError("reporter construction failed")

    result = CliRunner().invoke(main, ["start", "goal"])

    assert result.exit_code == EXIT_FAILURE
    assert "error: reporter construction failed" in result.stderr
    assert stubs.events == ["client", "orchestrator-ctor", "client.aclose"]
    assert "Traceback" not in result.output


def test_client_is_closed_even_when_the_error_is_unexpected(
    stubs: _Recorder,
    keyed: None,
) -> None:
    """Same guarantee for a bug in the constructor: it still propagates."""
    stubs.ctor_error = RuntimeError("reporter bug")

    result = CliRunner().invoke(main, ["start", "goal"])

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert stubs.events == ["client", "orchestrator-ctor", "client.aclose"]


# ── the --model regression ──────────────────────────────────────────────


def _run_with_models(
    monkeypatch: pytest.MonkeyPatch,
    stubs: _Recorder,
    args: list[str],
) -> _Recorder:
    """Invoke ``start`` with ``FURROW_MODEL=env-model`` and return the recorder."""
    monkeypatch.setenv("FURROW_ANTHROPIC_API_KEY", FAKE_API_KEY)
    monkeypatch.setenv("FURROW_MODEL", "env-model")

    with _global_settings_untouched():
        result = CliRunner().invoke(main, ["start", "goal", *args])

    assert result.exit_code == 0, result.output
    return stubs


def test_model_flag_overrides_env_for_every_role(
    monkeypatch: pytest.MonkeyPatch,
    stubs: _Recorder,
) -> None:
    """THE REGRESSION.

    ``--model`` used to set only ``model``, which no call site reads: the agents
    resolve through ``Settings.model_for(role)``, so the flag did literally
    nothing. It now sets all four fields, and the flag beats ``FURROW_MODEL``
    on every one of them.
    """
    _run_with_models(monkeypatch, stubs, ["--model", "flag-model"])

    assert stubs.models == {
        "model": "flag-model",
        "planner_model": "flag-model",
        "worker_model": "flag-model",
        "tester_model": "flag-model",
    }
    assert stubs.resolved.model_for("planner") == "flag-model"
    assert stubs.resolved.model_for("worker") == "flag-model"
    assert stubs.resolved.model_for("tester") == "flag-model"


def test_without_the_flag_the_env_model_wins_and_roles_keep_provider_defaults(
    monkeypatch: pytest.MonkeyPatch,
    stubs: _Recorder,
) -> None:
    """No flag: the env value is honoured, roles fall back to the provider default."""
    _run_with_models(monkeypatch, stubs, [])

    models = stubs.models
    assert models["model"] == "env-model"
    assert models["planner_model"] == PROVIDER_MODELS[Provider.ANTHROPIC]["planner_model"]
    assert models["planner_model"] != "env-model"


def test_role_specific_flag_beats_the_blanket_model(
    monkeypatch: pytest.MonkeyPatch,
    stubs: _Recorder,
) -> None:
    _run_with_models(
        monkeypatch,
        stubs,
        ["--model", "blanket", "--worker-model", "worker-only"],
    )

    assert stubs.models == {
        "model": "blanket",
        "planner_model": "blanket",
        "worker_model": "worker-only",
        "tester_model": "blanket",
    }


def test_role_specific_flag_works_without_the_blanket_model(
    monkeypatch: pytest.MonkeyPatch,
    stubs: _Recorder,
) -> None:
    """``--planner-model`` alone touches exactly one field and leaves the rest."""
    _run_with_models(monkeypatch, stubs, ["--planner-model", "solo-planner"])

    models = stubs.models
    assert models["planner_model"] == "solo-planner"
    assert models["model"] == "env-model", "model must be untouched"
    assert models["worker_model"] == PROVIDER_MODELS[Provider.ANTHROPIC]["worker_model"]
    assert models["tester_model"] == PROVIDER_MODELS[Provider.ANTHROPIC]["tester_model"]
    assert stubs.resolved.model_for("planner") == "solo-planner"


def test_start_does_not_mutate_the_global_settings_singleton(
    stubs: _Recorder,
    keyed: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second run with different flags must not move the singleton either."""
    from furrow.config import settings as global_settings

    snapshot = global_settings.model_dump()
    identity = id(global_settings)

    for args in (["--model", "first"], ["--model", "second"]):
        result = CliRunner().invoke(main, ["start", "goal", *args])
        assert result.exit_code == 0, result.output

    assert id(global_settings) == identity
    assert global_settings.model_dump() == snapshot
    # The per-run Settings really were distinct objects, so the check above is
    # not vacuous.
    assert stubs.resolved.model == "second"


def test_provider_flag_selects_a_different_provider_defaults(
    stubs: _Recorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--provider openai`` re-resolves every role model to the OpenAI default."""
    monkeypatch.setenv("FURROW_ANTHROPIC_API_KEY", FAKE_API_KEY)
    monkeypatch.setenv("FURROW_OPENAI_API_KEY", "sk-fake-OPENAI0001")

    result = CliRunner().invoke(main, ["start", "goal", "--provider", "openai"])

    assert result.exit_code == 0, result.output
    settings = stubs.resolved
    assert settings.provider is Provider.OPENAI
    assert settings.model == PROVIDER_MODELS[Provider.OPENAI]["model"]
    assert settings.planner_model == PROVIDER_MODELS[Provider.OPENAI]["planner_model"]
    assert settings.tester_model == PROVIDER_MODELS[Provider.OPENAI]["tester_model"]


def test_provider_flag_without_that_providers_key_is_an_error(stubs: _Recorder) -> None:
    """Switching provider without the matching credential must be refused."""
    result = CliRunner().invoke(main, ["start", "goal", "--provider", "openai"])

    assert result.exit_code == EXIT_FAILURE
    assert "error: provider 'openai' needs an API key but none is configured" in result.stderr
    assert "hint: set OPENAI_API_KEY, or FURROW_OPENAI_API_KEY" in result.stderr
    assert stubs.settings == [], "the run must not start"


def test_unknown_provider_is_a_usage_error() -> None:
    result = CliRunner().invoke(main, ["start", "goal", "--provider", "nope"])

    assert result.exit_code == EXIT_FAILURE
    assert "nope" in result.output


# ── packaging ───────────────────────────────────────────────────────────


def test_python_dash_m_furrow_help(tmp_path: Path) -> None:
    """``python -m furrow`` works without the console script being installed.

    Run from a directory that is not the repo root, with ``PYTHONPATH`` pointing
    at it: ``furrow`` is not pip-installed here, so without that the subprocess
    cannot import the package at all.
    """
    root = Path(__file__).resolve().parents[1]
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    env = {key: value for key, value in os.environ.items() if not key.startswith("FURROW_")}
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    env["PYTHONPATH"] = str(root)

    proc = subprocess.run(
        [sys.executable, "-m", "furrow", "--help"],
        cwd=elsewhere,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stderr
    assert "start" in proc.stdout
    assert "config" in proc.stdout
    assert "Traceback" not in proc.stdout
