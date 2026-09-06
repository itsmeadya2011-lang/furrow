from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=None, type=int, help="Override max cycles")
def start(goal: str | None, model: str | None, max_cycles: int | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    from furrow.config import Settings
    settings = Settings()
    if model:
        settings.model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    try:
        asyncio.run(Orchestrator(goal=goal, settings=settings).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
