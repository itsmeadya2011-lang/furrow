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
@click.option("--max-cycles", "max_cycles", default=0, show_default=True, help="Maximum planning cycles (0 = unlimited)")
@click.option("--max-parallel-tasks", "max_parallel_tasks", default=5, show_default=True, help="Maximum parallel tasks per cycle")
@click.option("--planner-model", "planner_model", default=None, help="Override planner model")
@click.option("--worker-model", "worker_model", default=None, help="Override worker model")
@click.option("--tester-model", "tester_model", default=None, help="Override tester model")
def start(
    goal: str | None,
    model: str | None,
    max_cycles: int,
    max_parallel_tasks: int,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    try:
        asyncio.run(
            Orchestrator(
                goal=goal,
                max_cycles=max_cycles,
                max_parallel_tasks=max_parallel_tasks,
                planner_model=planner_model,
                worker_model=worker_model,
                tester_model=tester_model,
            ).run()
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
