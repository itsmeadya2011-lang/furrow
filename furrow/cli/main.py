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
@click.option("--model", default=None, help="Override the model used by every agent")
@click.option(
    "--cycles",
    type=int,
    default=None,
    help="Stop after N cycles (0 = unbounded, the default)",
)
@click.option(
    "--max-parallel",
    "max_parallel",
    type=int,
    default=None,
    help="Cap concurrently executing tasks per cycle",
)
def start(
    goal: str | None,
    model: str | None,
    cycles: int | None,
    max_parallel: int | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        # Every agent reads its own model off the settings singleton, so the
        # override has to reach all of them, not just settings.model.
        settings.model = model
        settings.planner_model = model
        settings.worker_model = model
        settings.tester_model = model
    if cycles is not None:
        settings.max_cycles = max(0, cycles)
    if max_parallel is not None:
        settings.max_parallel_tasks = max(1, max_parallel)
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind address")
@click.option("--port", type=int, default=8000, show_default=True, help="Bind port")
@click.option(
    "--cycles",
    type=int,
    default=None,
    help="Cycle budget for every web session (0 = unbounded)",
)
@click.option(
    "--max-parallel",
    "max_parallel",
    type=int,
    default=None,
    help="Task fan-out cap for every web session",
)
def web(host: str, port: int, cycles: int | None, max_parallel: int | None) -> None:
    if cycles is not None:
        settings.max_cycles = max(0, cycles)
    if max_parallel is not None:
        settings.max_parallel_tasks = max(1, max_parallel)
    from furrow.web.server import run

    run(host=host, port=port)


if __name__ == "__main__":
    main()
