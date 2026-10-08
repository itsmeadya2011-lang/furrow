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
@click.option("--max-cycles", default=None, type=int, help="Maximum number of cycles")
@click.option("--resume", is_flag=True, help="Resume from saved state")
@click.option("--no-commit", is_flag=True, help="Disable auto-commit to git")
def start(goal: str | None, model: str | None, max_cycles: int | None, resume: bool, no_commit: bool) -> None:
    if not goal and not resume:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    try:
        asyncio.run(Orchestrator(
            goal=goal or "", 
            max_cycles=settings.max_cycles, 
            resume=resume,
            auto_commit=not no_commit
        ).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
