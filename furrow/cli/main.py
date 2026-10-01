from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
@click.option(
    "--log-level",
    default=None,
    help="Log level (DEBUG, INFO, WARNING, ERROR)",
    envvar="FURROW_LOG_LEVEL",
)
def main(log_level: str | None) -> None:
    """Furrow - autonomous coding agent with infinite parallel development loop."""
    if log_level:
        settings.log_level = log_level
    from furrow import configure_logging

    configure_logging(settings.log_level)


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
def start(goal: str | None, model: str | None) -> None:
    """Run Furrow's autonomous development loop for a goal."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def web(host: str, port: int) -> None:
    """Start the Furrow web UI."""
    from furrow.web.server import run

    run(host=host, port=port)


if __name__ == "__main__":
    main()
