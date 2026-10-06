from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--cycles", default=None, type=int, help="Max cycles to run (0 = unlimited)")
@click.option("--provider", default=None, type=click.Choice(["anthropic", "openai", "ollama"]))
def start(goal: str | None, model: str | None, cycles: int | None, provider: str | None) -> None:
    if goal is None:
        goal = click.prompt("Enter your goal for Furrow")
    if not goal.strip():
        console.print("[red]Error: goal cannot be empty.[/red]")
        sys.exit(1)
    from furrow.config import settings

    if model:
        settings.model = model
    if cycles is not None:
        settings.max_cycles = cycles
    if provider:
        settings.provider = provider
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
