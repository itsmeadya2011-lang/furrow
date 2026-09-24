from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import settings, configure_logging
from furrow.core.orchestrator import Orchestrator

configure_logging()

console = Console()


@click.group()
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), help="Log level")
@click.option("--log-format", default="console", type=click.Choice(["json", "console"]), help="Log format")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), help="LLM provider")
@click.option("--model", help="LLM model to use")
@click.option("--max-cycles", type=int, default=0, help="Maximum number of cycles (0 = unlimited)")
@click.option("--workspace", type=click.Path(exists=True, file_okay=False, dir_okay=True), help="Workspace directory")
def main(log_level: str, log_format: str, provider: str | None, model: str | None, max_cycles: int, workspace: str | None) -> None:
    settings.log_level = log_level
    settings.log_format = log_format
    if provider:
        settings.provider = provider
    if model:
        settings.model = model
    if max_cycles > 0:
        settings.max_cycles = max_cycles
    if workspace:
        settings.workspace = Path(workspace)


@main.command()
@click.argument("goal", required=False)
@click.option("--state-file", type=click.Path(), help="State file path")
def start(goal: str | None, state_file: str | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    try:
        asyncio.run(Orchestrator(goal=goal, state_file=state_file).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def web(host: str, port: int) -> None:
    from furrow.web.server import run
    run(host=host, port=port)


@main.command()
@click.argument("goal")
@click.option("--state-file", type=click.Path(), help="State file path")
def resume(goal: str, state_file: str | None) -> None:
    """Resume from a previous state file."""
    try:
        asyncio.run(Orchestrator(goal=goal, state_file=state_file).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
