from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import configure_logging, settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=None, type=int, help="Maximum development cycles (0 = infinite)")
@click.option("--max-parallel", default=None, type=int, help="Maximum parallel tasks")
@click.option("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)")
def start(goal: str | None, model: str | None, max_cycles: int | None, max_parallel: int | None, log_level: str) -> None:
    configure_logging(log_level)
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if max_parallel is not None:
        settings.max_parallel_tasks = max_parallel
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--port", default=8000, help="Port to bind")
def web(host: str, port: int) -> None:
    configure_logging("INFO")
    from furrow.web.server import run
    run(host=host, port=port)


if __name__ == "__main__":
    main()
