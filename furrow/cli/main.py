from __future__ import annotations

import asyncio
import logging
import sys

import click
from rich.console import Console

from furrow.config import configure_logging
from furrow.core.orchestrator import Orchestrator

console = Console()
logger = logging.getLogger(__name__)


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=None, type=int, help="Maximum number of development cycles")
def start(goal: str | None, model: str | None, max_cycles: int | None) -> None:
    configure_logging()
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    from furrow.config import settings

    if model:
        settings.model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    logger.info(f"Starting Furrow (goal={goal}, model={settings.model}, max_cycles={settings.max_cycles})")
    try:
        asyncio.run(Orchestrator(goal=goal, settings=settings).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--port", default=8000, type=int, help="Port to bind")
def web(host: str, port: int) -> None:
    configure_logging()
    from furrow.web.server import run

    run(host=host, port=port)


if __name__ == "__main__":
    main()
