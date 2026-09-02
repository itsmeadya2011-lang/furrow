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
@click.option("--model", default=None, help="Override default LLM model")
@click.option("--planner-model", default=None, help="Override planner model")
@click.option("--worker-model", default=None, help="Override worker model")
@click.option("--tester-model", default=None, help="Override tester model")
@click.option("--provider", default=None, help="Override LLM provider")
@click.option("--max-cycles", default=None, type=int, help="Maximum number of cycles (0 = unlimited)")
def start(
    goal: str | None,
    model: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    provider: str | None,
    max_cycles: int | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    from furrow.config import Settings, settings

    overrides: dict[str, object] = {}
    if model:
        overrides["model"] = model
    if planner_model:
        overrides["planner_model"] = planner_model
    if worker_model:
        overrides["worker_model"] = worker_model
    if tester_model:
        overrides["tester_model"] = tester_model
    if provider:
        overrides["provider"] = provider
    if max_cycles is not None:
        overrides["max_cycles"] = max_cycles

    cfg = settings.model_copy(update=overrides) if overrides else settings

    try:
        asyncio.run(Orchestrator(goal=goal, settings=cfg).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()