from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import Provider, settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=None, type=int, help="Maximum number of cycles (0 = unlimited)")
@click.option("--state-file", default=None, type=click.Path(), help="Path to state file for persistence")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="LLM provider")
def start(goal: str | None, model: str | None, max_cycles: int | None, state_file: str | None, provider: str | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if provider:
        settings.provider = Provider(provider)
    if state_file:
        state_path = Path(state_file)
    else:
        state_path = Path.cwd() / ".furrow_state.json"

    try:
        asyncio.run(Orchestrator(goal=goal, state_file=state_path).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
