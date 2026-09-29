from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model for all agents")
@click.option(
    "--provider",
    default=None,
    type=click.Choice(["anthropic", "openai", "ollama"], case_sensitive=False),
    help="Override LLM provider",
)
@click.option(
    "--max-cycles", default=None, type=int, help="Override max cycles (0 = unlimited)"
)
@click.option(
    "--max-parallel",
    default=None,
    type=int,
    help="Override max parallel tasks",
)
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    settings = Settings()
    if model:
        settings.model = model
        settings.planner_model = model
        settings.worker_model = model
        settings.tester_model = model
    if provider:
        from furrow.config import Provider

        settings.provider = Provider(provider)
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if max_parallel is not None:
        settings.max_parallel_tasks = max_parallel

    try:
        asyncio.run(Orchestrator(goal=goal, settings=settings).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--port", default=8000, type=int, help="Port to bind")
def web(host: str, port: int) -> None:
    from furrow.web.server import run

    run(host=host, port=port)


if __name__ == "__main__":
    main()
