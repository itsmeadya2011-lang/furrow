from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    """Furrow command-line interface."""
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
def start(goal: str | None, model: str | None) -> None:
    """Run Furrow to autonomously work toward a goal."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    try:
        orchestrator = Orchestrator(goal=goal)
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@main.command()
def web() -> None:
    """Run the Furrow web server."""
    from furrow.web.server import run

    try:
        run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow web server stopped.[/yellow]")
        sys.exit(0)


@main.command()
def serve() -> None:
    """Alias for the web command."""
    from furrow.web.server import run

    try:
        run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow web server stopped.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()