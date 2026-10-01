from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import Provider, settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model for all roles")
@click.option(
    "--provider",
    type=click.Choice([p.value for p in Provider]),
    default=None,
    help="Override LLM provider",
)
@click.option(
    "--max-cycles", type=int, default=None, help="Maximum planning cycles (0 = unlimited)"
)
@click.option("--max-parallel", type=int, default=None, help="Maximum parallel tasks")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
        settings.planner_model = model
        settings.worker_model = model
        settings.tester_model = model
    if provider:
        settings.provider = Provider(provider)
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
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", type=int, default=8000, help="Port to bind to")
def web(host: str, port: int) -> None:
    from furrow.web.server import run
    run(host=host, port=port)


if __name__ == "__main__":
    main()
