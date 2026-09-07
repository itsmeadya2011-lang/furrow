from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import get_logger
from furrow.core.orchestrator import Orchestrator

console = Console()
log = get_logger()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=0, type=int, help="Max development cycles (0 = unlimited)")
@click.option("--max-parallel", default=0, type=int, help="Max parallel worker tasks (0 = no limit)")
@click.option("--workspace", default=None, type=click.Path(), help="Working directory")
def start(goal: str | None, model: str | None, max_cycles: int, max_parallel: int, workspace: str | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    from furrow.config import settings

    if model:
        settings.model = model
    if max_cycles:
        settings.max_cycles = max_cycles
    if max_parallel:
        settings.max_parallel_tasks = max_parallel
    if workspace:
        settings.workspace = Path(workspace)
    try:
        log.info("cli.start", goal=goal, model=settings.model, max_cycles=settings.max_cycles)
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        log.info("cli.stopped_by_user")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
