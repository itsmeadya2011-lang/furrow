from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option(
    "--max-cycles", default=0, type=int, help="Maximum cycles before halting (0 = unlimited)"
)
def start(goal: str | None, model: str | None, max_cycles: int) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    try:
        asyncio.run(Orchestrator(goal=goal, max_cycles=max_cycles).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--port", default=8000, type=int, help="Port to listen on")
def web(host: str, port: int) -> None:
    from furrow.web.server import run

    run(host=host, port=port)


if __name__ == "__main__":
    main()
