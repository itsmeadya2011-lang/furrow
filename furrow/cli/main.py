from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow import configure_logging
from furrow.config import settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
@click.option("--log-level", default=None, help="Log level (DEBUG, INFO, WARNING, ERROR)")
def main(log_level: str | None) -> None:
    if log_level:
        settings.log_level = log_level
    configure_logging(settings.log_level)


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=None, type=int, help="Maximum number of cycles")
@click.option("--max-parallel", default=None, type=int, help="Maximum parallel tasks")
def start(goal: str | None, model: str | None, max_cycles: int | None, max_parallel: int | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if max_parallel is not None:
        settings.max_parallel_tasks = max_parallel
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()