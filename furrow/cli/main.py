from __future__ import annotations

import asyncio
import sys
from pathlib import Path

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
@click.option("--resume", is_flag=True, default=False, help="Resume from previous state")
@click.option("--clear-state", is_flag=True, default=False, help="Remove state file and start fresh")
def start(goal: str | None, model: str | None, resume: bool, clear_state: bool) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model

    state_file = settings.workspace / ".furrow_state.json"
    if clear_state and state_file.exists():
        state_file.unlink()
        console.print(f"[yellow]Cleared state file: {state_file}[/yellow]")

    try:
        asyncio.run(Orchestrator(goal=goal, state_file=state_file, resume=resume).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()