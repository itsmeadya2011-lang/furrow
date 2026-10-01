from __future__ import annotations

import asyncio
import sys

import click
import structlog
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

console = Console()

logger = structlog.get_logger(__name__)


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=0, help="Max development cycles (0 = unlimited)")
@click.option(
    "--provider",
    default=None,
    type=click.Choice(["anthropic", "openai", "ollama"]),
    help="Override LLM provider",
)
@click.option("--parallel", default=None, help="Max parallel tasks per cycle")
def start(
    goal: str | None,
    model: str | None,
    max_cycles: int,
    provider: str | None,
    parallel: int | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    from furrow.config import Provider, settings

    if model:
        settings.model = model
    if provider:
        settings.provider = Provider(provider)
    if parallel is not None:
        settings.max_parallel_tasks = parallel
    settings.max_cycles = max_cycles
    logger.info(
        "furrow_start",
        goal=goal,
        provider=settings.provider,
        max_cycles=settings.max_cycles,
    )
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
