from __future__ import annotations

import asyncio
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import click
from rich.console import Console

from furrow.config import Provider
from furrow.core.orchestrator import Orchestrator

try:
    __version__ = version("furrow")
except PackageNotFoundError:
    __version__ = "0.1.0"

console = Console()


@click.group()
@click.version_option(__version__)
def main() -> None:
    """Furrow - autonomous coding agent."""


@main.command()
@click.argument("goal", required=False)
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "ollama"]),
    default=None,
    help="LLM provider to use",
)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", type=int, default=None, help="Max planning cycles (0 = unlimited)")
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default=None,
    help="Logging level",
)
@click.option(
    "--state-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to the state file",
)
def start(
    goal: str | None,
    provider: str | None,
    model: str | None,
    max_cycles: int | None,
    log_level: str | None,
    state_file: Path | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    from furrow.config import settings

    if provider:
        settings.provider = Provider(provider)
    if model:
        settings.model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if log_level:
        settings.log_level = log_level
    if state_file is not None:
        settings.state_file = state_file

    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind the web server to")
@click.option("--port", type=int, default=8000, help="Port to bind the web server to")
def web(host: str, port: int) -> None:
    from furrow.web.server import run

    run(host=host, port=port)


if __name__ == "__main__":
    main()
