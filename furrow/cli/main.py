from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator
from furrow.logging import configure_logging

console = Console()


@click.group()
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), help="Log level")
@click.option("--log-file", default=None, type=click.Path(), help="Log file path")
@click.option("--workspace", default=None, type=click.Path(exists=True, file_okay=False), help="Workspace directory")
@click.option("--max-cycles", default=0, type=int, help="Maximum number of cycles (0 = unlimited)")
@click.option("--max-parallel", default=5, type=int, help="Maximum parallel tasks")
def main(log_level: str, log_file: str | None, workspace: str | None, max_cycles: int, max_parallel: int) -> None:
    """Furrow - Autonomous coding agent with infinite parallel development loop."""
    # Update settings from CLI
    settings.log_level = log_level
    settings.log_file = log_file
    if workspace:
        settings.workspace = workspace
    settings.max_cycles = max_cycles
    settings.max_parallel_tasks = max_parallel
    
    # Configure logging
    configure_logging(log_level, log_file)


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", default=None, type=click.Choice(["anthropic", "openai", "ollama"]), help="LLM provider")
def start(goal: str | None, model: str | None, provider: str | None) -> None:
    """Start Furrow with a goal."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if provider:
        settings.provider = provider
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--port", default=8000, type=int, help="Port to bind")
def web(host: str, port: int) -> None:
    """Start Furrow web UI."""
    from furrow.web.server import run
    run(host=host, port=port)


if __name__ == "__main__":
    main()