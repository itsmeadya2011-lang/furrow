"""Command-line entry point for Furrow.

Three commands: ``start`` runs the plan/execute/test loop on a goal, ``web``
serves the browser UI, and ``config`` prints the fully resolved settings so a
run can be inspected without starting one.

Two rules shape the implementation:

* **The CLI never mutates global state.** ``furrow.config.settings`` is a
  process-wide singleton; every command builds its own :class:`Settings` from
  :func:`_resolve_settings` instead of poking at it.
* **Expected failures are messages, not tracebacks.** A missing credential, an
  unparseable plan or a rate-limited provider should read as one actionable
  line, so :class:`~furrow.errors.FurrowError` is caught at the boundary and
  turned into ``error: ...`` on stderr plus exit code 2. Genuinely unexpected
  exceptions are left to propagate.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from enum import Enum
from pathlib import Path
from typing import Any, NoReturn

import click

from furrow import __version__
from furrow.config import Provider, Settings
from furrow.core.orchestrator import Orchestrator
from furrow.errors import FurrowError, MissingCredentialsError
from furrow.llm import LLMClient

_LOGGER = logging.getLogger("furrow")

#: Exit code for an expected failure: bad configuration, missing credentials,
#: an LLM or planning error. Matches click's own usage-error code.
EXIT_FAILURE = 2

#: 128 + SIGINT, the conventional status for "terminated by Ctrl-C". The old
#: ``sys.exit(0)`` claimed success for an interrupted, half-finished run.
EXIT_INTERRUPTED = 130

#: Reason handed to ``Orchestrator.request_stop`` on the first Ctrl-C.
INTERRUPT_REASON = "interrupted"

#: ``--model`` fans out to all four of these: the three agents resolve their
#: model through ``Settings.model_for(role)``, so setting only ``model`` would
#: leave the planner, worker and tester on whatever the provider default was.
MODEL_FIELDS = ("model", "planner_model", "worker_model", "tester_model")

#: Settings fields that hold a credential and must never be printed in the clear.
SECRET_FIELDS = frozenset({"anthropic_api_key", "openai_api_key"})

#: Settings field holding the credential a provider needs, plus the environment
#: variables accepted as a fallback. The fallbacks mirror ``furrow.llm``, which
#: reads the bare ``*_API_KEY`` variables too, so the CLI never refuses a
#: configuration the client itself would have accepted.
CREDENTIALS: dict[Provider, tuple[str, tuple[str, ...]]] = {
    Provider.ANTHROPIC: (
        "anthropic_api_key",
        ("FURROW_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    ),
    Provider.OPENAI: (
        "openai_api_key",
        ("FURROW_OPENAI_API_KEY", "OPENAI_API_KEY"),
    ),
}

#: ``validate_runtime()`` reports this; the CLI replaces it with a warning that
#: says where shell access came from. Matched by substring so a change of
#: wording in ``furrow.config`` degrades to printing both lines.
SUPERSEDED_WARNINGS = ("allow_shell is enabled",)

#: Environment variable holding the shared secret for the web UI. It is not a
#: ``Settings`` field, so it is read straight from the environment.
WEB_TOKEN_ENV = "FURROW_WEB_TOKEN"

#: Hosts that only bind interfaces reachable from this machine.
LOOPBACK_HOSTS = frozenset({"localhost", "::1", "127.0.0.1"})

#: Accepted ``--log-level`` values.
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

_PROVIDER_CHOICE = click.Choice([p.value for p in Provider], case_sensitive=False)
_LOG_LEVEL_CHOICE = click.Choice(LOG_LEVELS, case_sensitive=False)

_SHELL_PROMPT = (
    "Shell access lets the agent run arbitrary commands in the workspace. Enable it?"
)


# ── output helpers ──────────────────────────────────────────────────────


def _warn(message: str) -> None:
    """Print a non-fatal warning to stderr."""
    click.echo(f"warning: {message}", err=True)


def _fail(message: str, *, hint: str | None = None) -> NoReturn:
    """Print an error to stderr and exit with :data:`EXIT_FAILURE`."""
    click.echo(f"error: {message}", err=True)
    if hint:
        click.echo(f"hint: {hint}", err=True)
    raise SystemExit(EXIT_FAILURE)


def _report_failure(exc: FurrowError) -> NoReturn:
    """Turn an expected Furrow error into a clean message and exit code 2."""
    hint = exc.details.get("hint")
    _fail(str(exc), hint=hint if isinstance(hint, str) and hint else None)


# ── settings resolution ─────────────────────────────────────────────────


def _resolve_settings(**overrides: Any) -> Settings:
    """Build a Settings instance from the environment plus explicit overrides.

    Precedence is *CLI flag > ``FURROW_*`` env var > ``.env`` file > field
    default*, and it comes for free: pydantic-settings puts constructor
    keyword arguments ahead of every other source, so an override passed here
    always beats the environment.

    Callers must therefore pass ``None`` for options the user did not supply.
    ``None`` values are dropped here, so an absent flag never overrides an env
    var -- and the module-level ``furrow.config.settings`` singleton is left
    untouched.
    """
    return Settings(**{key: value for key, value in overrides.items() if value is not None})


def _model_overrides(
    model: str | None,
    *,
    planner: str | None = None,
    worker: str | None = None,
    tester: str | None = None,
) -> dict[str, str]:
    """Expand ``--model`` into every model the CLI knows how to set.

    ``--model`` is deliberately blunt: it sets ``model`` *and* every role model,
    so ``--model gpt-4o`` cannot leave the sub-agents pointing at Anthropic
    model ids. A role-specific flag wins over the blanket value, and a flag that
    was not passed contributes nothing, keeping the env layer intact. A
    role-specific flag works even when ``--model`` is not given.
    """
    overrides: dict[str, str] = {}
    if model:
        overrides.update(dict.fromkeys(MODEL_FIELDS, model))
    for field_name, specific in (
        ("planner_model", planner),
        ("worker_model", worker),
        ("tester_model", tester),
    ):
        if specific:
            overrides[field_name] = specific
    return overrides


def _missing_credential(settings: Settings) -> MissingCredentialsError | None:
    """The credential the selected provider needs but cannot find, if any."""
    entry = CREDENTIALS.get(settings.provider)
    if entry is None:
        # Ollama is a local server and needs no key.
        return None
    field_name, env_vars = entry
    if getattr(settings, field_name):
        return None
    if any(os.environ.get(name) for name in env_vars):
        return None
    return MissingCredentialsError(
        f"provider {settings.provider.value!r} needs an API key but none is configured",
        provider=settings.provider.value,
        hint=f"set {env_vars[-1]}, or {env_vars[0]} in your .env file",
    )


def _configure_logging(level: str | None, *, verbose: bool) -> None:
    """Send Furrow's logs to stderr at the requested level."""
    effective = "DEBUG" if verbose else (level or "INFO").upper()
    numeric = getattr(logging, effective, None)
    if not isinstance(numeric, int):
        numeric = logging.INFO
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # Set on the "furrow" logger rather than the root logger so an embedding
    # application's logging configuration is not replaced behind its back.
    logging.getLogger("furrow").setLevel(numeric)


# ── secrets ─────────────────────────────────────────────────────────────


def _mask_secret(value: str | None) -> str:
    """Render a credential as ``set (****abcd)``, never in the clear.

    A key of four characters or fewer is shown as ``set`` with no tail, because
    the "last 4 characters" of a short key *is* the key.
    """
    if not value:
        return "not set"
    if len(value) <= 4:
        return "set"
    return f"set (****{value[-4:]})"


def _render_setting(name: str, value: Any) -> str:
    """Render one settings field for ``config``, masking credentials."""
    if name in SECRET_FIELDS:
        return _mask_secret(value)
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


# ── goal and interrupt handling ─────────────────────────────────────────


def _resolve_goal(raw: str | None) -> str:
    """Return the goal to work on, prompting for it when none was passed."""
    text = raw
    if text is None:
        try:
            text = click.prompt("Enter the goal for Furrow", default="", show_default=False)
        except (click.Abort, EOFError):
            # No stdin to prompt with (a pipe, a CI job, a test harness).
            text = ""
    goal = (text or "").strip()
    if not goal:
        _fail('a goal is required: furrow start "<goal>"')
    return goal


def _is_loopback(host: str) -> bool:
    """Whether ``host`` binds only interfaces reachable from this machine."""
    candidate = host.strip().lower().strip("[]")
    return candidate in LOOPBACK_HOSTS or candidate.startswith("127.")


@contextmanager
def _stop_on_interrupt(orchestrator: Orchestrator) -> Iterator[None]:
    """Turn the first Ctrl-C into a graceful stop request for the loop.

    The loop notices ``request_stop`` between cycles and shuts down through its
    normal path, so the event stream still ends with a ``run_completed``. A
    second Ctrl-C hands the default disposition back, which keeps the escape
    hatch for a wedged cycle without raising from inside a loop callback.

    Platforms that cannot install loop signal handlers (a non-main thread, a
    restricted sandbox) simply fall through to the ``KeyboardInterrupt`` path in
    ``start``. The previous handler is always restored on the way out.
    """

    def _handle_sigint() -> None:
        hits[0] += 1
        if hits[0] == 1:
            orchestrator.request_stop(INTERRUPT_REASON)
            return
        _LOGGER.warning("second interrupt: aborting the run now")
        _restore()

    hits = [0]
    loop = asyncio.get_running_loop()
    previous: Any = None
    installed = False
    try:
        previous = signal.getsignal(signal.SIGINT)
        loop.add_signal_handler(signal.SIGINT, _handle_sigint)
        installed = True
    except (NotImplementedError, RuntimeError, ValueError):
        installed = False

    def _restore() -> None:
        if not installed:
            return
        with suppress(Exception):
            loop.remove_signal_handler(signal.SIGINT)
        if callable(previous):
            with suppress(Exception):
                signal.signal(signal.SIGINT, previous)

    try:
        yield
    finally:
        _restore()


async def _drive(goal: str, settings: Settings) -> None:
    """Run the loop for ``goal``, then always release the client and reporter.

    The client is built here and handed to the orchestrator, which leaves a
    caller-supplied client alone -- so closing it is this function's job, on the
    success path and on every failure path alike. Without it the httpx
    connection pools stay open and the interpreter complains at exit.
    """
    client = LLMClient(settings=settings)
    orchestrator: Orchestrator | None = None
    try:
        orchestrator = Orchestrator(goal=goal, client=client, settings=settings)
        with _stop_on_interrupt(orchestrator):
            await orchestrator.run()
    finally:
        # Orchestrator construction itself can raise (e.g. a misbehaving
        # reporter); guard with ``is not None`` so the client is always closed
        # and we never reference an unbound name.
        if orchestrator is not None:
            await orchestrator.aclose()
        await client.aclose()


# ── commands ────────────────────────────────────────────────────────────


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "--version", prog_name="furrow")
def main() -> None:
    """Furrow -- an autonomous coding agent that keeps working on a goal."""


@main.command()
@click.argument("goal", required=False)
@click.option(
    "--model",
    "model",
    default=None,
    metavar="NAME",
    help=(
        "Model for EVERY role (model, planner, worker and tester). Use this "
        "when you want one model everywhere; a role-specific flag below "
        "overrides it for that role."
    ),
)
@click.option(
    "--planner-model",
    default=None,
    metavar="NAME",
    help="Model for the planner agent; wins over --model.",
)
@click.option(
    "--worker-model",
    default=None,
    metavar="NAME",
    help="Model for the worker agent; wins over --model.",
)
@click.option(
    "--tester-model",
    default=None,
    metavar="NAME",
    help="Model for the tester agent; wins over --model.",
)
@click.option(
    "--provider",
    type=_PROVIDER_CHOICE,
    default=None,
    help="LLM provider. Defaults to the provider configured in the environment.",
)
@click.option(
    "--workspace",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory the agent may read and write.",
)
@click.option(
    "--max-cycles",
    type=click.IntRange(min=0),
    default=None,
    metavar="N",
    help="Stop after N cycles; 0 means run until the goal is done or it stalls.",
)
@click.option(
    "--max-parallel-tasks",
    type=click.IntRange(min=1),
    default=None,
    metavar="N",
    help="How many tasks may run at the same time in one cycle.",
)
@click.option(
    "--worker-iterations",
    type=click.IntRange(min=1),
    default=None,
    metavar="N",
    help="Tool-calling iterations a worker may spend on one task.",
)
@click.option(
    "--test-command",
    default=None,
    metavar="CMD",
    help="Command the tester runs to verify the work, e.g. 'pytest -q'.",
)
@click.option(
    "--allow-shell",
    is_flag=True,
    default=False,
    help=(
        "PRIVILEGE ESCALATION: let the agent run arbitrary shell commands in "
        "the workspace. Off by default; confirmed interactively, or requires "
        "--yes / --dangerously-allow-shell when there is no terminal."
    ),
)
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Assume yes for confirmation prompts. Required to enable --allow-shell "
    "in scripts and CI.",
)
@click.option(
    "--dangerously-allow-shell",
    is_flag=True,
    default=False,
    help="Enable shell access with --allow-shell without being asked to confirm.",
)
@click.option(
    "--log-level",
    type=_LOG_LEVEL_CHOICE,
    default=None,
    help="Furrow log level on stderr.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Shortcut for --log-level DEBUG.",
)
def start(
    goal: str | None,
    model: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    provider: str | None,
    workspace: Path | None,
    max_cycles: int | None,
    max_parallel_tasks: int | None,
    worker_iterations: int | None,
    test_command: str | None,
    allow_shell: bool,
    yes: bool,
    dangerously_allow_shell: bool,
    log_level: str | None,
    verbose: bool,
) -> None:
    """Work on GOAL until it is done, it stalls, or the cycle budget runs out.

    GOAL is prompted for when it is not given. Resolved settings come from the
    command line first, then FURROW_* environment variables, then .env.
    """
    try:
        _configure_logging(log_level, verbose=verbose)
        resolved_goal = _resolve_goal(goal)
        overrides: dict[str, Any] = {
            "provider": Provider(provider) if provider else None,
            "workspace": workspace,
            "max_cycles": max_cycles,
            "max_parallel_tasks": max_parallel_tasks,
            "worker_max_iterations": worker_iterations,
            "test_command": test_command,
            "log_level": log_level.upper() if log_level else None,
            "allow_shell": True if _confirm_shell(allow_shell, yes, dangerously_allow_shell) else None,
        }
        overrides.update(
            _model_overrides(
                model,
                planner=planner_model,
                worker=worker_model,
                tester=tester_model,
            )
        )
        settings = _resolve_settings(**overrides)

        for warning in settings.validate_runtime():
            if not warning.startswith(SUPERSEDED_WARNINGS):
                _warn(warning)
        if settings.allow_shell:
            source = "--allow-shell" if overrides.get("allow_shell") else "the environment"
            _warn(f"shell access is enabled via {source}: agents may run arbitrary commands")
        missing = _missing_credential(settings)
        if missing is not None:
            raise missing

        asyncio.run(_drive(resolved_goal, settings))
    except FurrowError as exc:
        _report_failure(exc)
    except KeyboardInterrupt:
        # Reached when the loop could not install its own SIGINT handler.
        _warn("interrupted; stopping")
        raise SystemExit(EXIT_INTERRUPTED) from None


def _confirm_shell(allow_shell: bool, yes: bool, dangerously: bool) -> bool:
    """Decide whether shell access is granted, and refuse to guess.

    ``--allow-shell`` alone is not enough: it must be confirmed at a terminal,
    or asserted with ``--yes`` / ``--dangerously-allow-shell`` for unattended
    runs. A ``--allow-shell`` in a pipeline with no terminal and no ``--yes``
    is an error rather than a silent grant.
    """
    if not allow_shell:
        return False
    if yes or dangerously:
        return True
    if sys.stdin.isatty():
        return click.confirm(_SHELL_PROMPT, default=False)
    _fail(
        "--allow-shell needs explicit confirmation: pass --yes or "
        "--dangerously-allow-shell, or run it in a terminal to be asked",
        hint="without it the agent cannot run shell commands, which is the safe default",
    )


@main.command()
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    metavar="ADDR",
    help="Interface to bind the UI to.",
)
@click.option(
    "--port",
    default=8000,
    show_default=True,
    type=click.IntRange(min=1, max=65535),
    help="TCP port to serve on.",
)
def web(host: str, port: int) -> None:
    """Serve the browser UI."""
    if not _is_loopback(host) and not os.environ.get(WEB_TOKEN_ENV):
        _warn(
            f"binding {host} exposes the Furrow UI to the network and {WEB_TOKEN_ENV} "
            "is not set: anyone who can reach this port can start runs against this "
            "machine. Set the token, or bind 127.0.0.1."
        )
    from furrow.web.server import run

    try:
        run(host=host, port=port)
    except FurrowError as exc:
        _report_failure(exc)
    except KeyboardInterrupt:
        _warn("web server stopped by user")
        raise SystemExit(EXIT_INTERRUPTED) from None


@main.command()
@click.option(
    "--workspace",
    type=click.Path(path_type=Path),
    default=None,
    help="Report this workspace instead of the configured one.",
)
@click.option(
    "--provider",
    type=_PROVIDER_CHOICE,
    default=None,
    help="Report this provider instead of the configured one.",
)
@click.option(
    "--model",
    default=None,
    metavar="NAME",
    help="Report this model for every role.",
)
def config(workspace: Path | None, provider: str | None, model: str | None) -> None:
    """Print the resolved settings, credentials masked."""
    overrides: dict[str, Any] = {
        "workspace": workspace,
        "provider": Provider(provider) if provider else None,
    }
    overrides.update(_model_overrides(model))
    settings = _resolve_settings(**overrides)

    fields = type(settings).model_fields
    width = max((len(name) for name in fields), default=0)
    for name in fields:
        click.echo(f"{name.ljust(width)}  {_render_setting(name, getattr(settings, name))}")

    for warning in settings.validate_runtime():
        if not warning.startswith(SUPERSEDED_WARNINGS):
            _warn(warning)
    missing = _missing_credential(settings)
    if missing is not None:
        _warn(str(missing))


if __name__ == "__main__":
    main()
