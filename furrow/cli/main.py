from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.core.orchestrator import Orchestrator
from furrow.logging import configure_logging

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=None, type=int, help="Maximum number of cycles (0 = unlimited)")
@click.option("--max-parallel", default=None, type=int, help="Maximum parallel worker tasks")
@click.option("--state-file", default=None, help="Path to state file for persistence")
@click.option("--persist", is_flag=True, default=False, help="Save and resume state from --state-file")
@click.option("--log-level", default="INFO", help="Logging level")
def start(
    goal: str | None,
    model: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
    state_file: str | None,
    persist: bool,
    log_level: str,
) -> None:
    configure_logging(log_level)
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    from furrow.config import Settings

    settings = Settings()
    if model:
        settings.model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if max_parallel is not None:
        settings.max_parallel_tasks = max_parallel
    if state_file:
        settings.state_file = state_file
    try:
        asyncio.run(Orchestrator(goal=goal, settings=settings, persist=persist).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default="8000")
@click.option("--log-level", default="INFO")
def web(host: str, port: int, log_level: str) -> None:
    from furrow.web.server import run

    run(host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
