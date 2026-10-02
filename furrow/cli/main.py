from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import Provider
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", type=int, default=None, help="Maximum agent cycles (0 = unlimited)")
@click.option(
    "--provider",
    type=click.Choice([p.value for p in Provider]),
    default=None,
    help="LLM provider to use",
)
def start(goal: str | None, model: str | None, max_cycles: int | None, provider: str | None) -> None:
    from furrow.config import settings

    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if provider:
        settings.provider = Provider(provider)
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run

    run()


if __name__ == "__main__":
    main()
