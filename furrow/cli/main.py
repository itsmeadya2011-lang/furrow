from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator
from furrow.git import GitManager

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option(
    "--max-cycles", default=None, type=int, help="Maximum number of cycles (0 = unlimited)"
)
@click.option("--no-git", is_flag=True, help="Disable git integration")
def start(goal: str | None, model: str | None, max_cycles: int | None, no_git: bool) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles

    git_manager = None if no_git else GitManager(Path(settings.workspace))

    try:
        asyncio.run(
            Orchestrator(goal=goal, max_cycles=settings.max_cycles, git_manager=git_manager).run()
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
