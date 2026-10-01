from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import Settings, setup_logging
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
@click.option("--log-level", default="INFO", help="Set log level (DEBUG, INFO, WARNING, ERROR)")
@click.pass_context
def main(ctx: click.Context, log_level: str) -> None:
    """Furrow: Autonomous coding agent with infinite parallel development loop."""
    setup_logging(log_level)
    ctx.ensure_object(dict)


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option(
    "--max-cycles",
    default=None,
    type=int,
    help="Maximum number of development cycles (0 = unlimited)",
)
@click.option("--max-parallel", default=None, type=int, help="Maximum parallel worker tasks")
@click.option(
    "--state-file", default=None, help="Path to persistent state file (relative to workspace)"
)
@click.pass_context
def start(
    ctx: click.Context,
    goal: str | None,
    model: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
    state_file: str | None,
) -> None:
    """Start the Furrow development loop with a GOAL."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    settings = Settings()
    if model:
        settings.model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if max_parallel is not None:
        settings.max_parallel_tasks = max_parallel
    if state_file:
        settings.state_file = state_file

    try:
        asyncio.run(Orchestrator(goal=goal, settings=settings).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--log-level", default="INFO", help="Log level")
def web(host: str, port: int, log_level: str) -> None:
    """Start the Furrow web UI server."""
    setup_logging(log_level)
    from furrow.web.server import run

    run(host=host, port=port)


if __name__ == "__main__":
    main()
