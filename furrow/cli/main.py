from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator
from furrow.logging import configure_logging

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=None, type=int, help="Cap development cycles (0 = unlimited)")
@click.option(
    "--parallel",
    default=None,
    type=int,
    help="Override max parallel worker tasks",
)
def start(
    goal: str | None, model: str | None, max_cycles: int | None, parallel: int | None
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    configure_logging(settings.log_level)
    if model:
        settings.model = model
    orchestrator = Orchestrator(
        goal=goal,
        max_cycles=max_cycles,
        max_parallel_tasks=parallel,
    )
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run

    configure_logging("INFO")
    run()


if __name__ == "__main__":
    main()
