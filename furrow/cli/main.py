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
@click.option("--max-cycles", default=0, type=int, help="Maximum cycles (0 = unlimited)")
@click.option("--max-parallel", default=None, type=int, help="Max parallel tasks per cycle")
@click.option("--workspace", default=None, help="Workspace directory")
def start(
    goal: str | None,
    model: str | None,
    max_cycles: int,
    max_parallel: int | None,
    workspace: str | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    from furrow.config import Settings, settings

    overrides: dict = {}
    if model:
        overrides["model"] = model
    if max_cycles:
        overrides["max_cycles"] = max_cycles
    if max_parallel is not None:
        overrides["max_parallel_tasks"] = max_parallel
    if workspace:
        overrides["workspace"] = workspace

    run_settings = settings
    if overrides:
        run_settings = Settings(**{**settings.model_dump(), **overrides})

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
