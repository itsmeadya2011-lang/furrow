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
def start(goal: str | None, model: str | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        from furrow.config import settings
        settings.model = model
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


@main.command()
@click.option("--state-file", default=".furrow_state.json", help="Path to state file")
def resume(state_file: str) -> None:
    from furrow.llm import LLMClient
    loaded = Orchestrator.load_state(state_file, client=LLMClient())
    if loaded is None:
        console.print("[red]No saved state found. Run a goal first.[/red]")
        sys.exit(1)
    try:
        asyncio.run(loaded.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
