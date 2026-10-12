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
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", type=int, default=None, help="Maximum planning cycles")
@click.option("--parallel", type=int, default=None, help="Maximum parallel tasks")
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "ollama"]),
    default=None,
    help="LLM provider",
)
@click.option("--timeout", type=int, default=None, help="Request timeout in seconds")
def start(
    goal: str | None,
    model: str | None,
    max_cycles: int | None,
    parallel: int | None,
    provider: str | None,
    timeout: int | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if parallel is not None:
        settings.max_parallel_tasks = parallel
    if provider is not None:
        settings.provider = Provider(provider)
    if timeout is not None:
        settings.request_timeout = timeout
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
