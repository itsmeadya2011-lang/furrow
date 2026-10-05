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
@click.option(
    "--provider",
    default=None,
    type=click.Choice([p.value for p in Provider]),
    help="Override LLM provider",
)
@click.option("--max-cycles", default=0, type=int, help="Max planning cycles (0=unlimited)")
@click.option("--parallel", default=None, type=int, help="Override max parallel tasks")
@click.option("--state-file", default=None, type=click.Path(path_type=str), help="State file path")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int,
    parallel: int | None,
    state_file: str | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if provider:
        settings.provider = Provider(provider)
    if parallel is not None:
        settings.max_parallel_tasks = parallel
    if state_file:
        settings.state_file = state_file
    try:
        orchestrator = Orchestrator(goal=goal)
        asyncio.run(orchestrator.run(max_cycles=max_cycles))
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run

    run()


if __name__ == "__main__":
    main()
