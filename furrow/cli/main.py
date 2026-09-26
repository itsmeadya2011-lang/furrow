from __future__ import annotations

import asyncio
import logging
import sys

import click
import structlog
from rich.console import Console

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
    help="Override max cycles (0 = unlimited)",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose (DEBUG) output",
)
def start(
    goal: str | None,
    model: str | None,
    max_cycles: int | None,
    verbose: bool,
) -> None:
    from furrow.config import settings

    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    if model:
        settings.model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if verbose:
        settings.verbose = True
        settings.log_level = "DEBUG"
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
        structlog.configure()

    try:
        asyncio.run(Orchestrator(goal=goal, verbose=verbose).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run

    run()


if __name__ == "__main__":
    main()
