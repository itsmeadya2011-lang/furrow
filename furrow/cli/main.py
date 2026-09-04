from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow import __version__
from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

console = Console()
settings = Settings()


@click.group()
@click.version_option(version=__version__, prog_name="furrow")
def main() -> None:
    """Furrow command-line interface."""


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model used for orchestration.")
def start(goal: str | None, model: str | None) -> None:
    """Start an interactive Furrow orchestration session.

    If GOAL is omitted, Furrow will prompt for it interactively.
    """
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    try:
        asyncio.run(Orchestrator(goal=goal, settings=settings).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Host interface to bind the web server to.")
@click.option("--port", default=8000, show_default=True, type=int, help="Port to bind the web server to.")
def web(host: str, port: int) -> None:
    """Launch the Furrow web UI and WebSocket server."""
    from furrow.web.server import run
    run(host=host, port=port)


if __name__ == "__main__":
    main()
