"""Command line interface for furrow.

Three properties the CLI is built around:

* **No global mutation.**  ``furrow.config.settings`` is process-wide state that
  outlives a command (and is read by every ``LLMClient`` built later in the same
  interpreter), so ``start`` builds a *fresh* :class:`~furrow.config.Settings`
  from the singleton's values plus the flag overrides and hands that to the
  orchestrator.  Pydantic validates the new instance, which is also what turns a
  bad ``--max-cycles`` into a message instead of a traceback.
* **``--model`` really means something.**  Each agent passes its own role model
  explicitly (planner, worker, tester), so the old ``settings.model = model``
  assignment changed nothing that ever ran.  The flag now sets every role's model.
* **Expected failures are messages, not tracebacks**, and an interrupted run
  exits 130 (128 + SIGINT) rather than reporting success to the shell.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

import click
from pydantic import ValidationError
from rich.console import Console

#: Set when importing the configuration or the orchestrator failed; see the guard.
_CONFIGURATION_ERROR: ValueError | None

try:
    # ``furrow.config`` builds its ``Settings`` singleton at import time, so a
    # malformed ``FURROW_*`` / ``.env`` value makes importing *it* raise - which,
    # because these names are read at decoration and call time, would make
    # importing this module raise too.  Both imports therefore live inside the
    # guard: the failure has to be deferred to command execution so the user gets
    # an explanation instead of a pydantic traceback out of ``config.py`` before
    # the prompt even appears.  ``ValidationError`` is a ``ValueError`` subclass.
    from furrow.config import MODEL_ROLE_FIELDS, Provider, Settings
    from furrow.config import settings as global_settings
    from furrow.core.orchestrator import Orchestrator
except ValueError as exc:  # pragma: no cover - only reachable with a broken environment
    _CONFIGURATION_ERROR = exc
    # Inert placeholders so the module stays importable and every name is still
    # defined for the type checker.  No command reaches them: ``start`` re-runs
    # the guarded import and reports the real error before using any of them.
    MODEL_ROLE_FIELDS = ()  # type: ignore[assignment,misc]
    Provider = ()  # type: ignore[assignment,misc]
    Settings = None  # type: ignore[assignment,misc]
    global_settings = None  # type: ignore[assignment,misc]
    Orchestrator = None  # type: ignore[assignment,misc]
else:
    _CONFIGURATION_ERROR = None

console = Console()

#: Exit code for an expected failure (bad configuration, missing credentials).
#: Click already uses 2 for a usage error such as an unknown flag.
EXIT_ERROR = 1

#: Exit code for a run the user interrupted: 128 + SIGINT, the shell convention.
#: Exiting 0 would tell CI that the work succeeded.
EXIT_INTERRUPTED = 130

#: Distribution whose metadata ``--version`` reports.
_DISTRIBUTION = "furrow"


def _version() -> str:
    """The installed *furrow* version, or a marker when it is not installed.

    Read from the installed distribution metadata rather than a second copy of the
    string in this file, and never by parsing ``pyproject.toml`` at runtime.  A
    plain source checkout has no metadata, which click's own detection turns into
    a ``RuntimeError``, so that case degrades to a readable marker instead.
    """
    try:
        return distribution_version(_DISTRIBUTION)
    except PackageNotFoundError:
        return "unknown (furrow is not installed; this looks like a source checkout)"


@click.group()
@click.version_option(
    version=_version(),
    prog_name=_DISTRIBUTION,
    message="%(prog)s %(version)s",
)
def main() -> None:
    """Autonomous coding agent: plan, implement, test - until the goal holds."""


# --------------------------------------------------------------------------- start


def _prompt_for_goal() -> str:
    """Ask for a goal on stdin.

    ``default=""`` is what makes this usable in CI: without it click re-prompts
    forever on empty input from a pipe instead of failing.  ``click.prompt``
    turns EOF and Ctrl-C into :class:`click.Abort`, which is reported as a usage
    problem rather than a traceback.
    """
    try:
        return str(click.prompt("Enter your goal for furrow", default="", show_default=False))
    except (click.Abort, EOFError):
        raise click.ClickException(
            "No goal given and stdin is not interactive. "
            'Pass the goal as an argument: furrow start "your goal".'
        ) from None


def _resolve_goal(goal: str | None) -> str:
    """Return a non-empty goal, prompting for one when it was not given."""
    candidate = goal if goal is not None and goal.strip() else _prompt_for_goal()
    cleaned = candidate.strip()
    if not cleaned:
        raise click.ClickException(
            "The goal is empty. Pass it as an argument: furrow start \"your goal\"."
        )
    return cleaned


def _model_overrides(model: str) -> dict[str, str]:
    """``--model`` overrides *every* role's model, not the unused fallback.

    ``PlannerAgent`` completes with ``settings.planner_model``, the worker with
    ``worker_model`` and the tester with ``tester_model``, and ``Settings.model``
    is only ever a fallback inside ``LLMClient.complete()`` that no caller
    reaches.  Setting ``model`` alone therefore changed nothing observable, so
    the flag sets all of them together.
    """
    return dict.fromkeys(MODEL_ROLE_FIELDS, model)


def _validation_summary(exc: ValidationError) -> str:
    """Flatten a pydantic error into ``field: message`` pairs on one line."""
    parts: list[str] = []
    for error in exc.errors():
        field = ".".join(str(part) for part in error["loc"]) or "(settings)"
        parts.append(f"{field}: {error['msg']}")
    summary = "; ".join(parts[:5])
    if len(parts) > 5:
        summary += f"; and {len(parts) - 5} more"
    return summary or str(exc)


def _configuration_error(exc: ValueError) -> str:
    """Turn an expected startup failure into an actionable one-line message."""
    if isinstance(exc, ValidationError):
        return (
            f"Invalid configuration ({_validation_summary(exc)}). "
            "Check the FURROW_* environment variables and the .env file."
        )
    message = str(exc)
    if "API_KEY" in message:
        return (
            f"Cannot start furrow: {message}. Set the provider's key in the "
            "environment or in a .env file (FURROW_ANTHROPIC_API_KEY or "
            "ANTHROPIC_API_KEY, FURROW_OPENAI_API_KEY or OPENAI_API_KEY)."
        )
    return f"Cannot start furrow: {message}"


def _effective_settings(overrides: dict[str, Any]) -> Settings:
    """Build a fresh ``Settings`` from the singleton's values plus *overrides*.

    The singleton is copied, never mutated: it is read by anything else running in
    this interpreter, and assigning onto it would both leak the override and skip
    nothing - pydantic still validates every field of the new instance, which is
    what rejects an out-of-range ``--max-cycles`` before any LLM call happens.
    """
    values: dict[str, Any] = global_settings.model_dump()
    values.update(overrides)
    return Settings(**values)


async def _run_and_close(orchestrator: Orchestrator) -> None:
    """Run the loop and release the LLM client's HTTP sessions either way."""
    try:
        await orchestrator.run()
    finally:
        # ``Orchestrator.aclose`` forwards to ``LLMClient.aclose``, which is
        # idempotent.  This also runs when Ctrl-C cancels the task, which is the
        # only path that would otherwise leak open sockets.
        with contextlib.suppress(Exception):
            await orchestrator.aclose()


def _run(orchestrator: Orchestrator) -> None:
    """Drive one run, mapping Ctrl-C to a 130 exit code."""
    try:
        asyncio.run(_run_and_close(orchestrator))
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        # If the interruption landed between the loop starting and the task's
        # ``finally`` running, close in a fresh loop; ``aclose`` is idempotent.
        with contextlib.suppress(Exception):
            asyncio.run(orchestrator.aclose())
        sys.exit(EXIT_INTERRUPTED)


@main.command()
@click.argument("goal", required=False)
@click.option(
    "--provider",
    type=click.Choice([member.value for member in Provider]),
    default=None,
    help="LLM provider to use, instead of the configured one.",
)
@click.option(
    "--model",
    default=None,
    metavar="MODEL_ID",
    help="Model id used by every role (planner, workers and tester), replacing "
    "the per-role defaults.",
)
@click.option(
    "--max-parallel-tasks",
    type=int,
    default=None,
    metavar="N",
    help="Maximum number of worker tasks to run at once (>= 1).",
)
@click.option(
    "--max-cycles",
    type=int,
    default=None,
    metavar="N",
    help="Stop after N plan/execute/test cycles (>= 0). 0 means unlimited, which "
    "is the default.",
)
@click.option(
    "--workspace",
    type=click.Path(path_type=Path),
    default=None,
    metavar="DIR",
    help="Directory furrow is allowed to work in.",
)
def start(
    goal: str | None,
    provider: str | None,
    model: str | None,
    max_parallel_tasks: int | None,
    max_cycles: int | None,
    workspace: Path | None,
) -> None:
    """Run furrow until GOAL is satisfied.

    Anything the flag values cannot express - API keys, per-provider defaults,
    timeouts - comes from the environment or a .env file, as documented in the
    README.
    """
    objective = _resolve_goal(goal)

    overrides: dict[str, Any] = {}
    if provider is not None:
        overrides["provider"] = provider
    if model is not None:
        overrides.update(_model_overrides(model))
    if max_parallel_tasks is not None:
        overrides["max_parallel_tasks"] = max_parallel_tasks
    if max_cycles is not None:
        overrides["max_cycles"] = max_cycles
    if workspace is not None:
        overrides["workspace"] = workspace

    orchestrator_class = Orchestrator
    if orchestrator_class is None:
        # The guarded import at the top of this module failed, so the
        # configuration cannot be loaded.  Re-run that import to raise the real
        # validation error here, where it becomes a message with an exit code.
        try:
            from furrow.core.orchestrator import Orchestrator as orchestrator_class
        except ValueError as exc:
            raise click.ClickException(_configuration_error(exc)) from exc

    try:
        # Validated here, so an out-of-range --max-cycles or --max-parallel-tasks is
        # reported before the first LLM call.
        settings = _effective_settings(overrides)
        orchestrator = orchestrator_class(goal=objective, settings=settings, console=console)
    except ValueError as exc:
        # Missing credentials, an unusable provider, or a malformed value from the
        # environment.  ``ValidationError`` is a ``ValueError`` subclass.
        raise click.ClickException(_configuration_error(exc)) from exc

    _run(orchestrator)


# ---------------------------------------------------------------------------- web


def _is_loopback(host: str) -> bool:
    """True when *host* is reachable from this machine only."""
    if host in ("localhost", "localhost.localdomain"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:  # a hostname, not an address literal
        return False


@main.command()
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    metavar="ADDRESS",
    help="Address to bind. The web UI has NO authentication and lets anyone who "
    "reaches it drive an agent that edits files and runs shell commands, so keep "
    "it on loopback or put an authenticating proxy in front of it.",
)
@click.option(
    "--port",
    type=click.IntRange(1, 65535),
    default=8000,
    show_default=True,
    metavar="PORT",
    help="TCP port to listen on.",
)
def web(host: str, port: int) -> None:
    """Serve the web UI and its WebSocket event stream (unauthenticated)."""
    try:
        from furrow.web.server import run
    except ValueError as exc:
        # Same deferred-import contract as ``start``: a broken ``FURROW_*`` /
        # ``.env`` value reaches the user as a message, not a traceback.
        raise click.ClickException(_configuration_error(exc)) from exc

    if not _is_loopback(host):
        console.print(
            f"[yellow]Warning: binding {host} exposes an unauthenticated agent that "
            "edits files and runs shell commands to everything that can reach "
            f"{host}:{port}.[/yellow]"
        )
    # Keyword arguments, so this survives a change to the server's defaults.
    run(host=host, port=port)


if __name__ == "__main__":
    main()