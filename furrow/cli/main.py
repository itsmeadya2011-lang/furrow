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
@click.option("--model", default=None, help="Override LLM model for all agents")
@click.option("--planner-model", default=None, help="Override LLM model for the planner agent")
@click.option("--worker-model", default=None, help="Override LLM model for the worker agent")
@click.option("--tester-model", default=None, help="Override LLM model for the tester agent")
@click.option("--max-cycles", default=None, type=int, help="Override maximum number of cycles")
@click.option(
    "--max-parallel", default=None, type=int, help="Override maximum number of parallel tasks"
)
def start(
    goal: str | None,
    model: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if (
        model
        or planner_model
        or worker_model
        or tester_model
        or max_cycles is not None
        or max_parallel is not None
    ):
        from furrow.config import settings

        if model:
            settings.model = model
            settings.planner_model = model
            settings.worker_model = model
            settings.tester_model = model
        if planner_model:
            settings.planner_model = planner_model
        if worker_model:
            settings.worker_model = worker_model
        if tester_model:
            settings.tester_model = tester_model
        if max_cycles is not None:
            settings.max_cycles = max_cycles
        if max_parallel is not None:
            settings.max_parallel_tasks = max_parallel
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
