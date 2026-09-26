from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import settings, setup_logging
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
    type=int,
    default=None,
    help="Override maximum number of cycles (0 = unlimited)",
)
@click.option(
    "--workspace",
    type=click.Path(path_type=Path),
    default=None,
    help="Override workspace directory",
)
@click.option(
    "--concurrency",
    type=int,
    default=None,
    help="Override maximum parallel tasks",
)
def start(
    goal: str | None,
    model: str | None,
    max_cycles: int | None,
    workspace: Path | None,
    concurrency: int | None,
) -> None:
    setup_logging()
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
        settings.worker_model = model
        settings.tester_model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if concurrency is not None:
        settings.max_parallel_tasks = concurrency
    if workspace is not None:
        settings.workspace = workspace
        os.chdir(workspace)
    try:
        asyncio.run(Orchestrator(goal=goal, console=console).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run

    run()


if __name__ == "__main__":
    main()
