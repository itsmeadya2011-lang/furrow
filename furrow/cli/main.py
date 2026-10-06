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
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="Override LLM provider")
@click.option("--max-cycles", type=int, default=None, help="Maximum number of cycles")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), default=None, help="Log level")
def start(goal: str | None, model: str | None, provider: str | None, max_cycles: int | None, log_level: str | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if provider:
        from furrow.config import Provider
        settings.provider = Provider(provider)
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if log_level:
        settings.log_level = log_level
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
