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
@click.option("--max-cycles", default=0, help="Maximum number of cycles (0 = unlimited)")
@click.option("--workspace", default=None, type=click.Path(exists=True, file_okay=False, path_type=Path), help="Workspace directory")
@click.option("--resume", is_flag=True, help="Resume from previous state")
def start(goal: str | None, model: str | None, max_cycles: int, workspace: Path | None, resume: bool) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        from furrow.config import settings
        settings.model = model
    ws = workspace or Path.cwd()
    try:
        asyncio.run(Orchestrator(goal=goal, max_cycles=max_cycles, workspace=ws, resume=resume).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
