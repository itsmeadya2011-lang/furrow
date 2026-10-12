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
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=0, help="Maximum cycles (0 = unlimited)")
@click.option("--max-parallel", default=None, type=int, help="Max parallel tasks")
def start(goal: str | None, model: str | None, max_cycles: int, max_parallel: int | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    
    settings = Settings()
    if model:
        settings.model = model
    if max_cycles > 0:
        settings.max_cycles = max_cycles
    if max_parallel is not None:
        settings.max_parallel_tasks = max_parallel
    
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
