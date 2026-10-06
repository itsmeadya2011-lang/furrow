from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
import structlog
from rich.console import Console

from furrow.config import Settings, settings
from furrow.core.orchestrator import Orchestrator

console = Console()


def configure_logging(log_level: str) -> None:
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
    )


@click.group()
@click.option("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)")
def main(log_level: str) -> None:
    configure_logging(log_level)


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--state-file", default=None, help="Path to state file for persistence")
@click.option("--max-cycles", default=0, help="Maximum cycles (0 = unlimited)")
def start(goal: str | None, model: str | None, state_file: str | None, max_cycles: int) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    if model:
        settings.model = model
    if max_cycles:
        settings.max_cycles = max_cycles

    try:
        asyncio.run(
            Orchestrator(
                goal=goal,
                settings_override=settings,
                state_file=state_file,
            ).run()
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run

    run()


@main.command()
@click.argument("state_file", default=".furrow_state.json")
def resume(state_file: str) -> None:
    """Resume from a saved state file."""
    path = Path(state_file)
    if not path.exists():
        console.print(f"[red]State file not found: {state_file}[/red]")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    goal = data.get("goal", "")
    if not goal:
        console.print("[red]No goal found in state file[/red]")
        sys.exit(1)

    console.print(f"[green]Resuming goal: {goal}[/green]")

    try:
        asyncio.run(Orchestrator(goal=goal, state_file=state_file).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()