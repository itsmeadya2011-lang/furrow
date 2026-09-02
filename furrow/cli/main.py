from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

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
    type=click.Choice(["anthropic", "openai", "ollama"]),
    default=None,
    help="LLM provider",
)
@click.option(
    "--max-parallel",
    type=int,
    default=None,
    help="Max parallel tasks",
)
@click.option(
    "--max-cycles",
    type=int,
    default=None,
    help="Max planning cycles",
)
@click.option(
    "--workspace",
    type=click.Path(file_okay=False),
    default=None,
    help="Working directory",
)
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_parallel: int | None,
    max_cycles: int | None,
    workspace: str | None,
) -> None:
    from furrow.config import settings

    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if provider:
        settings.provider = provider
    if max_parallel is not None:
        settings.max_parallel_tasks = max_parallel
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if workspace:
        settings.workspace = workspace
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
