from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import Settings, settings
from furrow.core.orchestrator import Orchestrator
from furrow.logging import configure_logging

console = Console()


@click.group()
@click.option("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)")
@click.option("--log-json/--no-log-json", default=False, help="Output logs as JSON")
def main(log_level: str, log_json: bool) -> None:
    configure_logging(level=log_level, json_output=log_json)
    settings.log_level = log_level
    settings.log_json = log_json


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=0, help="Maximum number of cycles (0 = unlimited)")
@click.option("--max-parallel", default=5, help="Maximum parallel tasks")
def start(goal: str | None, model: str | None, max_cycles: int, max_parallel: int) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    settings.max_cycles = max_cycles
    settings.max_parallel_tasks = max_parallel
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def web(host: str, port: int) -> None:
    from furrow.web.server import run
    run(host=host, port=port)


if __name__ == "__main__":
    main()