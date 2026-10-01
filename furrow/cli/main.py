from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

console = Console()


def _build_client(model: str | None, max_cycles: int | None = None):
    """Create an LLMClient with optional settings overrides.

    When ``model`` is provided, all per-role model fields are set to it
    so that the planner, worker, and tester all use the overridden model.
    When ``max_cycles`` is provided, the cycle limit is overridden.
    """
    from furrow.config import Settings, settings as global_settings
    from furrow.llm import LLMClient

    base = global_settings.model_dump()
    if model:
        base["model"] = model
        base["planner_model"] = model
        base["worker_model"] = model
        base["tester_model"] = model
    if max_cycles is not None:
        base["max_cycles"] = max_cycles
    settings = Settings(**base)
    client = LLMClient(settings=settings)
    return settings, client


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=None, type=int, help="Limit the number of development cycles")
def start(goal: str | None, model: str | None, max_cycles: int | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    settings, client = _build_client(model, max_cycles)
    try:
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
