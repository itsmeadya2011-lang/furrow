from __future__ import annotations

import asyncio
import sys
from typing import Any

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
@click.option("--model", default=None, help="Override LLM model")
@click.option("--cycles", type=int, default=None, help="Maximum number of cycles (0 = unlimited)")
@click.option("--max-parallel", type=int, default=None, help="Maximum parallel tasks per cycle")
def start(goal: str | None, model: str | None, cycles: int | None, max_parallel: int | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    # Build settings with CLI overrides
    overrides: dict[str, Any] = {}
    if model:
        overrides["model"] = model
    if cycles is not None:
        overrides["max_cycles"] = cycles
    if max_parallel is not None:
        overrides["max_parallel_tasks"] = max_parallel

    custom_settings = Settings(**overrides)
    from furrow.llm import LLMClient
    client = LLMClient(settings=custom_settings)

    try:
        asyncio.run(Orchestrator(goal=goal, client=client, settings=custom_settings).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
