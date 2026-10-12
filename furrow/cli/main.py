from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import configure_logging, settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
@click.option("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)")
def main(log_level: str) -> None:
    configure_logging(log_level)


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=0, type=int, help="Maximum number of cycles (0 = unlimited)")
@click.option("--max-parallel", default=None, type=int, help="Maximum parallel tasks")
@click.option("--resume", is_flag=True, help="Resume from saved state")
def start(goal: str | None, model: str | None, max_cycles: int, max_parallel: int | None, resume: bool) -> None:
    if not goal and not resume:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if max_cycles > 0:
        settings.max_cycles = max_cycles
    if max_parallel is not None:
        settings.max_parallel_tasks = max_parallel

    try:
        asyncio.run(Orchestrator(goal=goal or "", resume=resume).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
def web(host: str, port: int) -> None:
    from furrow.web.server import run

    run(host=host, port=port)


if __name__ == "__main__":
    main()