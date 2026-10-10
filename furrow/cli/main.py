from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import setup_logging
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
@click.option("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)")
def main(log_level: str) -> None:
    setup_logging(log_level)


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--state-file", default=None, help="State file for persistence")
def start(goal: str | None, model: str | None, state_file: str | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        from furrow.config import settings
        settings.model = model
    try:
        asyncio.run(Orchestrator(goal=goal, state_file=state_file).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
