from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    """Furrow CLI entry point."""


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
def start(goal: str | None, model: str | None) -> None:
    """Start a Furrow run toward a given goal.

    If no goal is provided, the user is prompted interactively.
    When ``--model`` is supplied, a local ``Settings`` instance is created
    so the global singleton is never mutated.
    """
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    local_settings = Settings(model=model) if model else None
    try:
        asyncio.run(Orchestrator(goal=goal, settings=local_settings).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    """Start the Furrow web interface."""
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
