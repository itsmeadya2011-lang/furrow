from __future__ import annotations

import asyncio
import sys
from pathlib import Path

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
@click.option("--cycles", default=None, type=int, help="Max cycles to run (0 = unlimited)")
@click.option("--workspace", default=None, type=click.Path(path_type=Path), help="Workspace directory")
def start(goal: str | None, model: str | None, cycles: int | None, workspace: Path | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    from furrow.config import settings
    if model:
        settings.model = model
    if cycles is not None:
        settings.max_cycles = cycles
    if workspace is not None:
        settings.workspace = Path(workspace).resolve()
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
