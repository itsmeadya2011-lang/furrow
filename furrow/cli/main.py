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
@click.option("--planner-model", default=None, help="Override planner model")
@click.option("--worker-model", default=None, help="Override worker model")
@click.option("--tester-model", default=None, help="Override tester model")
@click.option("--max-cycles", default=None, type=int, help="Max cycles (0 = unlimited)")
@click.option("--provider", default=None, type=click.Choice(["anthropic", "openai", "ollama"]), help="LLM provider")
def start(
    goal: str | None,
    model: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    max_cycles: int | None,
    provider: str | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    from furrow.config import settings
    if model:
        settings.model = model
    if planner_model:
        settings.planner_model = planner_model
    if worker_model:
        settings.worker_model = worker_model
    if tester_model:
        settings.tester_model = tester_model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
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
