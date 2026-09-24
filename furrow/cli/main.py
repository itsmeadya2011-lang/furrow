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
@click.option("--max-cycles", default=0, help="Maximum number of cycles (0 = unlimited)")
def start(goal: str | None, model: str | None, max_cycles: int) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        from furrow.config import settings
        settings.model = model
    if max_cycles > 0:
        from furrow.config import settings
        settings.max_cycles = max_cycles
    
    orchestrator = Orchestrator(goal=goal)
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
    finally:
        orchestrator.print_stats()
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()