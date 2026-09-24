from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
import structlog
from rich.console import Console

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

console = Console()


def setup_logging(log_level: str) -> None:
    """Configure structlog for structured logging."""
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(log_level.upper()),
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
            if sys.stderr.isatty()
            else structlog.processors.JSONRenderer(),
        ],
    )


@click.group()
@click.option("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)")
def main(log_level: str) -> None:
    setup_logging(log_level)


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=0, type=int, help="Maximum number of cycles (0 = unlimited)")
@click.option(
    "--state-file",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to state file for persistence",
)
@click.option("--resume", is_flag=True, help="Resume from state file")
def start(
    goal: str | None, model: str | None, max_cycles: int, state_file: Path | None, resume: bool
) -> None:
    if resume and state_file and state_file.exists():
        console.print(f"[green]Resuming from {state_file}[/green]")
        try:
            orchestrator = Orchestrator.load_state(state_file)
            if model:
                orchestrator.settings.model = model
            if max_cycles > 0:
                orchestrator.settings.max_cycles = max_cycles
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            console.print(f"[red]Failed to load state: {e}[/red]")
            sys.exit(1)
    else:
        if not goal:
            goal = click.prompt("Enter your goal for Furrow")

        settings = Settings()
        if model:
            settings.model = model
        if max_cycles > 0:
            settings.max_cycles = max_cycles

        orchestrator = Orchestrator(goal=goal, settings=settings)

    # Set state file for auto-saving
    if state_file:
        orchestrator._state_file = state_file

    try:
        asyncio.run(orchestrator.run())

        # Save state on completion
        if state_file:
            orchestrator.save_state(state_file)

    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        if state_file:
            orchestrator.save_state(state_file)
            console.print(f"[dim]State saved to {state_file}[/dim]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run

    run()


if __name__ == "__main__":
    main()
