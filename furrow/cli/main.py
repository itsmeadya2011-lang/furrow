from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import Settings, settings
from furrow.core.orchestrator import Orchestrator
from furrow.logging_config import setup_logging

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="LLM provider")
@click.option("--max-cycles", default=None, type=int, help="Maximum number of cycles")
@click.option("--log-level", default=None, type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), help="Log level")
@click.option("--resume", is_flag=True, help="Resume from previous state")
def start(goal: str | None, model: str | None, provider: str | None, max_cycles: int | None, log_level: str | None, resume: bool) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    
    # Apply CLI overrides to settings
    if model:
        settings.model = model
    if provider:
        settings.provider = provider
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if log_level:
        settings.log_level = log_level
    
    # Setup logging
    setup_logging(settings)
    
    try:
        asyncio.run(Orchestrator(goal=goal, resume=resume).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def web(host: str, port: int) -> None:
    from furrow.web.server import run
    setup_logging(settings)
    run(host=host, port=port)


if __name__ == "__main__":
    main()
