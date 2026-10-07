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
@click.option("--max-cycles", default=10, help="Maximum number of cycles")
@click.option("--resume", is_flag=True, help="Resume from checkpoint")
def start(goal: str | None, model: str | None, max_cycles: int, resume: bool) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        from furrow.config import settings
        settings.model = model
    try:
        asyncio.run(Orchestrator(goal=goal, max_cycles=max_cycles, resume=resume).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
