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
@click.option("--cycles", type=int, default=None, help="Override max cycles")
@click.option("--max-parallel", type=int, default=None, help="Override max parallel tasks")
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "ollama"]),
    default=None,
    help="Override LLM provider",
)
def start(
    goal: str | None,
    model: str | None,
    cycles: int | None,
    max_parallel: int | None,
    provider: str | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if cycles is not None:
        settings.max_cycles = cycles
    if max_parallel is not None:
        settings.max_parallel_tasks = max_parallel
    if provider is not None:
        settings.provider = provider
    try:
        asyncio.run(Orchestrator(goal=goal, settings=settings).run())
    except ValueError as exc:
        if "API_KEY" in str(exc):
            console.print(
                "\n[red]No API key configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY "
                "in your environment (or FURROW_ANTHROPIC_API_KEY / "
                "FURROW_OPENAI_API_KEY).[/red]"
            )
            sys.exit(1)
        raise
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()