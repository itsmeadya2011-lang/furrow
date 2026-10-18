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
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option(
    "--max-cycles",
    default=None,
    type=int,
    help="Maximum number of cycles before halting (0 = unlimited)",
)
@click.option(
    "--max-parallel-tasks",
    default=None,
    type=int,
    help="Maximum number of tasks to run in parallel",
)
def start(
    goal: str | None,
    model: str | None,
    max_cycles: int | None,
    max_parallel_tasks: int | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if max_parallel_tasks is not None:
        settings.max_parallel_tasks = max_parallel_tasks
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
