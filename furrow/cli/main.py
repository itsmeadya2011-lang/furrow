from __future__ import annotations

import asyncio
import os
import sys

import click
from rich.console import Console

from furrow.config import Provider, configure_logging, settings
from furrow.core.orchestrator import Orchestrator

console = Console()


def _check_api_key() -> bool:
    """Check if the required API key is configured for the current provider."""
    if settings.provider == Provider.ANTHROPIC:
        if not settings.anthropic_api_key and not os.getenv("ANTHROPIC_API_KEY"):
            console.print(
                "[red]Error: ANTHROPIC_API_KEY is not set. "
                "Set it via env or .env file.[/red]"
            )
            return False
    elif settings.provider == Provider.OPENAI:
        if not settings.openai_api_key and not os.getenv("OPENAI_API_KEY"):
            console.print(
                "[red]Error: OPENAI_API_KEY is not set. "
                "Set it via env or .env file.[/red]"
            )
            return False
    return True


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model (applies to all agents)")
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "ollama"]),
    default=None,
    help="Override LLM provider",
)
@click.option("--cycles", type=int, default=None, help="Override max cycles")
@click.option("--parallel", type=int, default=None, help="Override max parallel tasks")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    cycles: int | None,
    parallel: int | None,
) -> None:
    configure_logging(settings.log_level)
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
        settings.planner_model = model
        settings.worker_model = model
        settings.tester_model = model
    if provider:
        settings.provider = Provider(provider)
    if cycles is not None:
        settings.max_cycles = cycles
    if parallel is not None:
        settings.max_parallel_tasks = parallel
    if not _check_api_key():
        sys.exit(1)
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    configure_logging(settings.log_level)
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
