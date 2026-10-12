from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import settings
from furrow.core.logging import configure_logging
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
@click.option("--log-level", default=None, help="Log level (DEBUG, INFO, WARNING, ERROR)")
@click.option("--log-file", default=None, type=click.Path(path_type=Path), help="Log file path")
def main(log_level: str | None, log_file: Path | None) -> None:
    configure_logging(log_level or settings.log_level, log_file)


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=0, help="Maximum number of cycles (0 = unlimited)")
@click.option("--resume", is_flag=True, help="Resume from previous state")
@click.option("--state-dir", default=None, type=click.Path(exists=True, file_okay=False, path_type=Path), help="State directory (default: current workspace)")
def start(goal: str | None, model: str | None, max_cycles: int, resume: bool, state_dir: Path | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    try:
        asyncio.run(Orchestrator(goal=goal, max_cycles=max_cycles, resume=resume, state_dir=state_dir).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
