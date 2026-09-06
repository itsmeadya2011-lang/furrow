from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import Settings, setup_logging
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--workspace", default=None, help="Working directory")
def start(goal: str | None, model: str | None, workspace: str | None) -> None:
    settings = Settings()
    setup_logging(settings.log_level)
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if workspace:
        settings.workspace = Path(workspace)
    try:
        asyncio.run(Orchestrator(goal=goal, settings=settings).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--workspace", default=None, help="Working directory")
def web(workspace: str | None) -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
