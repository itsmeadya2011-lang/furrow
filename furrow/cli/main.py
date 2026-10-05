"""Command line entry point for Furrow.

``main`` is the console script declared in ``pyproject.toml``
(``furrow = furrow.cli.main:main``) and must keep that name.  Two commands are
offered:

``furrow start GOAL``
    Runs one goal in the terminal.  Every option maps onto a validated
    :class:`furrow.config.Settings` field, the resolved options are applied to a
    *new* settings object (never to the process-global singleton), and the exit
    code says what happened: ``0`` the goal was completed, ``1`` the run halted
    for another reason (stop requested, cycle budget exhausted, planner error),
    ``130`` the run was interrupted with Ctrl-C, ``2`` invalid usage.

``furrow web``
    Serves the web UI.  Binds ``127.0.0.1`` unless ``--host`` says otherwise,
    because a Furrow run can write files in the workspace.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import sys
from pathlib import Path
from typing import Any, NoReturn

import click
import structlog
from pydantic import ValidationError
from rich.console import Console

from furrow.config import LOG_LEVELS, Settings, configure_logging
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

console = Console()
log = structlog.get_logger(__name__)

EXIT_COMPLETED = 0
EXIT_HALTED = 1
EXIT_INTERRUPTED = 130

UNKNOWN_VERSION = "unknown"
"""Used by ``--version`` when the package is not installed (e.g. from a checkout)."""

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def package_version() -> str:
    """The installed ``furrow`` version, or ``unknown`` when it is not installed.

    ``importlib.metadata`` is the single source of truth: the version lives in
    ``pyproject.toml`` and is never duplicated here.
    """
    try:
        return importlib.metadata.version("furrow")
    except importlib.metadata.PackageNotFoundError:
        return UNKNOWN_VERSION


def _describe_validation_error(exc: ValidationError) -> str:
    """Flatten a pydantic error into one readable line for click."""
    parts: list[str] = []
    for item in exc.errors():
        location = ".".join(str(piece) for piece in item.get("loc", ()) if piece != "__root__")
        message = str(item.get("msg", "invalid value"))
        parts.append(f"{location}: {message}" if location else message)
    return "; ".join(parts) or "invalid value"


def _usage_error(message: str) -> NoReturn:
    """Report a bad option value as a usage error instead of a traceback."""
    raise click.UsageError(message)


def _build_settings(**overrides: Any) -> Settings:
    """Validate the CLI overrides through :class:`Settings`.

    ``Settings`` re-reads ``FURROW_*`` env vars and ``.env``, so only the values
    the user actually passed are overridden and everything still goes through
    the pydantic validators.
    """
    try:
        return Settings(**overrides)
    except ValidationError as exc:
        _usage_error(f"invalid configuration: {_describe_validation_error(exc)}")


def _model_overrides(
    model: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
) -> dict[str, str]:
    """Resolve the four model fields.

    Precedence: the per-agent options set their own field, and ``--model``
    overrides all four afterwards, so ``--model`` always wins.
    """
    overrides: dict[str, str] = {}
    for value, field_name in (
        (planner_model, "planner_model"),
        (worker_model, "worker_model"),
        (tester_model, "tester_model"),
    ):
        if value is not None:
            overrides[field_name] = value
    if model is not None:
        overrides.update(
            model=model,
            planner_model=model,
            worker_model=model,
            tester_model=model,
        )
    return overrides


def _resolve_goal(goal: str | None) -> str:
    """Use the positional argument, or prompt for a goal when it is missing."""
    resolved = goal.strip() if goal else ""
    if not resolved:
        resolved = click.prompt("Enter your goal for Furrow").strip()
    if not resolved:
        console.print("[red]No goal given; nothing to do.[/red]")
        sys.exit(EXIT_HALTED)
    return resolved


async def _drive(goal: str, run_settings: Settings, client: LLMClient) -> bool:
    """Run one goal, always closing *client* before returning.

    A cancellation is re-raised after a graceful ``stop()`` so the loop unwinds
    in order, and ``client.aclose()`` runs in the ``finally`` so httpx
    connections are released on both the happy and the interrupted path.
    """
    orchestrator = Orchestrator(
        goal=goal,
        client=client,
        settings=run_settings,
        on_event=_log_event,
    )
    try:
        return await orchestrator.run()
    except asyncio.CancelledError:
        await orchestrator.stop()
        raise
    finally:
        with contextlib.suppress(Exception):
            await client.aclose()


def _log_event(event: str, message: str) -> None:
    """Mirror orchestrator events into the structured log.

    ``event`` is the first positional parameter of a structlog bound logger, so
    the orchestrator's event name is logged under ``name``.
    """
    log.info("orchestrator_event", name=event, message=message)


def run_goal(goal: str, run_settings: Settings) -> int:
    """Run *goal* to completion and map the outcome onto an exit code.

    ``asyncio.run`` cancels and awaits the run task when ``KeyboardInterrupt``
    surfaces inside the loop, so no worker task survives the interrupt.
    """
    client = LLMClient(settings=run_settings)
    try:
        completed = asyncio.run(_drive(goal, run_settings, client))
    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Interrupted: the run was stopped and its tasks cancelled.[/yellow]"
        )
        return EXIT_INTERRUPTED
    if completed:
        console.print("[bold green]Goal complete.[/bold green]")
        return EXIT_COMPLETED
    console.print("[bold yellow]Furrow halted before the goal was met.[/bold yellow]")
    return EXIT_HALTED


@click.group()
@click.version_option(
    package_version(),
    "--version",
    "-V",
    prog_name="furrow",
    message="%(prog)s %(version)s",
)
def main() -> None:
    """Furrow: an autonomous coding agent with a parallel development loop."""


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, metavar="NAME", help="Model for every agent.")
@click.option("--planner-model", default=None, metavar="NAME", help="Model for the planner.")
@click.option("--worker-model", default=None, metavar="NAME", help="Model for the workers.")
@click.option("--tester-model", default=None, metavar="NAME", help="Model for the tester.")
@click.option(
    "--provider", default=None, metavar="NAME", help="LLM backend: anthropic, openai or ollama."
)
@click.option(
    "--max-parallel-tasks",
    default=None,
    type=int,
    metavar="N",
    help="Tasks run concurrently (>= 1).",
)
@click.option(
    "--max-cycles", default=None, type=int, metavar="N", help="Cycle budget (0 = unlimited)."
)
@click.option(
    "--workspace",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    metavar="DIR",
    help="Directory furrow reads and writes.",
)
@click.option("--test-command", default=None, metavar="CMD", help="Explicit test command.")
@click.option(
    "--log-level",
    default=None,
    type=click.Choice(LOG_LEVELS, case_sensitive=False),
    help="Minimum log level.",
)
def start(
    goal: str | None,
    model: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    provider: str | None,
    max_parallel_tasks: int | None,
    max_cycles: int | None,
    workspace: Path | None,
    test_command: str | None,
    log_level: str | None,
) -> None:
    """Run Furrow on GOAL (prompted for when omitted).

    Model precedence: --planner-model/--worker-model/--tester-model set their own
    agent, and --model then overrides all four (including the planner and tester
    defaults, which --model alone used to leave untouched).
    """
    overrides: dict[str, Any] = _model_overrides(model, planner_model, worker_model, tester_model)
    if provider is not None:
        overrides["provider"] = provider
    if max_parallel_tasks is not None:
        overrides["max_parallel_tasks"] = max_parallel_tasks
    if max_cycles is not None:
        overrides["max_cycles"] = max_cycles
    if workspace is not None:
        overrides["workspace"] = workspace
    if test_command is not None:
        overrides["test_command"] = test_command
    if log_level is not None:
        overrides["log_level"] = log_level

    run_settings = _build_settings(**overrides)
    configure_logging(run_settings.log_level)
    resolved_goal = _resolve_goal(goal)
    console.print(f"[dim]provider={run_settings.provider} model={run_settings.model}[/dim]")
    sys.exit(run_goal(resolved_goal, run_settings))


@main.command()
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    metavar="ADDR",
    help=(
        "Interface to bind. The default keeps runs private to this machine; "
        "0.0.0.0 exposes an agent that can write files to the network."
    ),
)
@click.option("--port", default=8000, show_default=True, type=int, metavar="PORT", help="TCP port.")
@click.option(
    "--log-level",
    default=None,
    type=click.Choice(LOG_LEVELS, case_sensitive=False),
    help="Minimum log level.",
)
@click.option(
    "--max-cycles",
    default=None,
    type=int,
    metavar="N",
    help=(
        "Cycle budget for runs started in the browser. 0 keeps the web default "
        "(8); values above the web hard cap (50) are clamped."
    ),
)
@click.option(
    "--max-concurrent-runs",
    default=None,
    type=int,
    metavar="N",
    help="Runs served at the same time (>= 1).",
)
def web(
    host: str,
    port: int,
    log_level: str | None,
    max_cycles: int | None,
    max_concurrent_runs: int | None,
) -> None:
    """Serve the web UI (default http://127.0.0.1:8000).

    Set FURROW_WEB_TOKEN to require a token for every run; without it, only a
    loopback bind keeps the agent private.
    """
    if not 1 <= port <= 65535:
        _usage_error(f"--port must be between 1 and 65535, got {port}")
    if max_cycles is not None and max_cycles < 0:
        _usage_error(f"--max-cycles must be 0 (web default) or more, got {max_cycles}")
    if max_concurrent_runs is not None and max_concurrent_runs < 1:
        _usage_error(f"--max-concurrent-runs must be at least 1, got {max_concurrent_runs}")

    from furrow.web.server import run as run_server
    from furrow.web.server import web_token_configured

    overrides: dict[str, Any] = {}
    if max_cycles is not None:
        overrides["max_cycles"] = max_cycles
    if log_level is not None:
        overrides["log_level"] = log_level
    run_settings = _build_settings(**overrides)
    configure_logging(run_settings.log_level)
    if host not in LOOPBACK_HOSTS and not web_token_configured():
        console.print(
            "[bold yellow]warning:[/bold yellow] binding a non-loopback interface without "
            "FURROW_WEB_TOKEN: anyone who can reach this port can start an agent that "
            "writes files in the workspace."
        )
    run_server(
        host=host,
        port=port,
        settings=run_settings,
        log_level=run_settings.log_level,
        max_concurrent_runs=max_concurrent_runs,
    )


if __name__ == "__main__":
    main()
