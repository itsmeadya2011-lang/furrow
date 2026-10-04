"""Command line entry point for Furrow."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from rich.console import Console

if TYPE_CHECKING:
    from furrow.config import Settings

console = Console()

LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"


def _settings() -> Settings:
    """Import the settings singleton lazily, so `--help` stays cheap."""
    from furrow.config import get_settings

    return get_settings()


def _configure_logging() -> None:
    """Honour `settings.log_level`, which nothing else configures."""
    level = _settings().log_level
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
    else:
        logging.basicConfig(level=level, format=LOG_FORMAT, stream=sys.stderr)
        root.setLevel(level)


def _print_version(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    from furrow import __version__

    click.echo(f"furrow {__version__}")
    ctx.exit()


def _effective_settings(
    model: str | None,
    provider: str | None,
    workspace: Path | None,
    max_cycles: int | None,
) -> Settings:
    """Return a Settings copy with the CLI overrides applied.

    The global singleton is never mutated: every agent takes an explicit model, so
    `--model` has to reach planner/worker/tester to have any effect at all.
    """
    from furrow.config import Provider

    base = _settings()
    updates: dict[str, Any] = {}
    if model:
        updates.update(
            model=model,
            planner_model=model,
            worker_model=model,
            tester_model=model,
        )
    if provider:
        try:
            updates["provider"] = Provider(provider)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="--provider") from exc
    if workspace is not None:
        resolved = workspace.expanduser()
        if not resolved.is_dir():
            raise click.BadParameter(
                f"{resolved} is not an existing directory", param_hint="--workspace"
            )
        updates["workspace"] = resolved
    if max_cycles is not None:
        if max_cycles < 0:
            raise click.BadParameter("must be 0 or more", param_hint="--max-cycles")
        updates["max_cycles"] = max_cycles
    if not updates:
        return base
    return base.model_copy(update=updates)


@click.group()
@click.option(
    "--version",
    is_flag=True,
    expose_value=False,
    is_eager=True,
    callback=_print_version,
    help="Show the Furrow version and exit.",
)
def main() -> None:
    """Furrow - an autonomous coding agent that plans, builds and tests in parallel.

    Run a goal with `furrow start GOAL`, or drive it from a browser with `furrow web`.
    Defaults come from FURROW_* environment variables or a .env file, and every one of
    them can be overridden per command.
    """
    _configure_logging()


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Model for every agent (overrides FURROW_*).")
@click.option("--provider", default=None, help="LLM provider: anthropic or openai.")
@click.option(
    "--workspace",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory the agent may read and write.",
)
@click.option(
    "--max-cycles",
    "max_cycles",
    default=None,
    type=int,
    help="Stop after this many planning cycles (0 means no limit).",
)
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    workspace: Path | None,
    max_cycles: int | None,
) -> None:
    """Work on GOAL until Furrow decides it is done. Press Ctrl-C to stop."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    goal = goal.strip()
    if not goal:
        raise click.UsageError("goal must not be empty")

    settings = _effective_settings(model, provider, workspace, max_cycles)
    console.print(
        f"[dim]provider={settings.provider.value} model={settings.model} "
        f"workspace={settings.workspace} max_cycles={settings.max_cycles}[/dim]"
    )

    from furrow.core.orchestrator import Orchestrator

    orchestrator = Orchestrator(goal=goal, settings=settings, console=console)
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        # asyncio.run has already cancelled the run task; ask the orchestrator to stop
        # cooperatively too so any partially written file is left alone.
        orchestrator.cancel()
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(130)


@main.command()
@click.option("--host", default=None, help="Interface to bind (default 127.0.0.1).")
@click.option("--port", default=None, type=int, help="Port to bind (default 8000).")
def web(host: str | None, port: int | None) -> None:
    """Serve the web UI on loopback. There is no authentication."""
    settings = _settings()
    resolved_host = settings.web_host if host is None else host
    resolved_port = settings.web_port if port is None else port
    if not 1 <= resolved_port <= 65535:
        raise click.BadParameter("must be between 1 and 65535", param_hint="--port")

    from furrow.web.server import run

    try:
        run(host=resolved_host, port=resolved_port)
    except KeyboardInterrupt:
        console.print("\n[yellow]Web server stopped by user.[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
