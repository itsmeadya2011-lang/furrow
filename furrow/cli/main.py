from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model for all agents")
@click.option("--cycles", default=None, type=int, help="Max development cycles (0 = unlimited)")
def start(goal: str | None, model: str | None, cycles: int | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    overrides: dict = {}
    if model:
        overrides.update(
            model=model,
            planner_model=model,
            worker_model=model,
            tester_model=model,
        )
    if cycles is not None:
        overrides["max_cycles"] = cycles

    settings = Settings(**overrides)

    try:
        client = LLMClient(settings=settings)
        asyncio.run(Orchestrator(goal=goal, client=client).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run

    run()


if __name__ == "__main__":
    main()
