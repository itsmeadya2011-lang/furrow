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
@click.option("--resume", is_flag=True, default=False, help="Resume from saved state")
def start(goal: str | None, model: str | None, resume: bool) -> None:
    orchestrator = Orchestrator(goal=goal or "")
    if resume:
        if orchestrator.load_state():
            console.print("[green]Resumed from saved state.[/green]")
        else:
            console.print("[yellow]No saved state found. Starting fresh.[/yellow]")
    if model:
        from furrow.config import settings
        settings.model = model
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
