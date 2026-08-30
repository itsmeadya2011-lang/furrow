from __future__ import annotations

import asyncio
import logging
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
@click.option("--planner-model", default=None, help="Override planner model")
@click.option("--worker-model", default=None, help="Override worker model")
@click.option("--max-cycles", default=0, type=int, help="Maximum orchestration cycles (0 = unlimited)")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def start(goal: str | None, model: str | None, planner_model: str | None, worker_model: str | None, max_cycles: int, verbose: bool) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    if model:
        from furrow.config import settings
        settings.model = model
    if planner_model:
        from furrow.config import settings
        settings.planner_model = planner_model
    if worker_model:
        from furrow.config import settings
        settings.worker_model = worker_model
    try:
        asyncio.run(Orchestrator(goal=goal, max_cycles=max_cycles).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
