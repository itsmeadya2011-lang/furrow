from __future__ import annotations

import asyncio
import logging
import sys

import click
from rich.console import Console
from structlog import get_logger

from furrow.core.orchestrator import Orchestrator
from furrow.config import settings

console = Console()
logger = get_logger(__name__)


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=0, help="Max cycles before halting (0=unlimited)")
def start(goal: str | None, model: str | None, max_cycles: int) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    settings.max_cycles = max_cycles
    logger.info("start_command", goal=goal, model=model, max_cycles=max_cycles)
    try:
        asyncio.run(Orchestrator(goal=goal, client=None).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
