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
@click.option("--state-file", default=None, help="Path to state file to resume from")
def resume(state_file: str | None) -> None:
    from furrow.config import settings
    from furrow.llm import LLMClient
    if state_file:
        settings.state_file = Path(state_file)
    client = LLMClient(settings=settings)
    orch = Orchestrator(goal="", client=client, settings=settings)
    if not orch.goal:
        orch.goal = click.prompt("Enter your goal for Furrow")
    if orch.cycles > 0:
        console.print(f"[yellow]Resuming from previous run ({orch.cycles} cycles completed)[/yellow]")
    try:
        asyncio.run(orch.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
