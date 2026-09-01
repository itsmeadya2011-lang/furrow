from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import configure_logging
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
def start(goal: str | None, model: str | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        from furrow.config import settings
        settings.model = model
    configure_logging()
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    configure_logging()
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
