from __future__ import annotations

import asyncio
import json
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
@click.option("--max-cycles", default=None, type=int, help="Maximum number of cycles")
@click.option("--state-file", default=".furrow_state.json", help="State file path")
def start(goal: str | None, model: str | None, max_cycles: int | None, state_file: str) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles

    state_path = Path(state_file)
    existing_state = {}
    if state_path.exists():
        try:
            existing_state = json.loads(state_path.read_text())
            if not goal and existing_state.get("goal"):
                goal = existing_state["goal"]
                console.print(f"[cyan]Resuming goal: {goal}[/cyan]")
        except Exception:  # noqa: BLE001, S110
            pass

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