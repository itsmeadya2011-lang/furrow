from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import Settings, Provider
from furrow.core.orchestrator import Orchestrator
from furrow.logging import get_logger

console = Console()
_logger = get_logger("cli")


@click.group()
def main() -> None:
    """Furrow: autonomous coding agent with an infinite parallel development loop."""
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=None, type=int, help="Maximum number of cycles (0 = unlimited)")
@click.option("--max-parallel", default=None, type=int, help="Maximum parallel tasks per cycle")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None,
              help="LLM provider to use")
def start(goal: str | None, model: str | None, max_cycles: int | None,
          max_parallel: int | None, provider: str | None) -> None:
    """Start the Furrow development loop with the given goal."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    # Build overrides for settings
    overrides: dict = {}
    if model:
        overrides["model"] = model
    if max_cycles is not None:
        overrides["max_cycles"] = max_cycles
    if max_parallel is not None:
        overrides["max_parallel_tasks"] = max_parallel
    if provider:
        overrides["provider"] = Provider(provider)

    from furrow.config import settings as default_settings
    cli_settings = Settings(**{**default_settings.model_dump(), **overrides}) if overrides else default_settings

    _logger.info("Starting orchestrator", goal=goal, provider=cli_settings.provider, model=cli_settings.model)
    try:
        asyncio.run(Orchestrator(goal=goal, settings=cli_settings).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        _logger.info("Stopped by user")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
def web(host: str, port: int) -> None:
    """Start the Furrow web UI server."""
    from furrow.web.server import run
    _logger.info("Starting web server", host=host, port=port)
    run(host=host, port=port)


if __name__ == "__main__":
    main()
