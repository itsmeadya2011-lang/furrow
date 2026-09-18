from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import settings as global_settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
def start(goal: str | None, model: str | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    run_settings = global_settings
    if model:
        run_settings = global_settings.model_copy(
            update={
                "model": model,
                "planner_model": model,
                "worker_model": model,
                "tester_model": model,
            }
        )
    try:
        asyncio.run(Orchestrator(goal=goal, settings=run_settings).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
