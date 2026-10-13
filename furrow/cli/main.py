from __future__ import annotations

import asyncio
import sys

import click
import structlog
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

console = Console()

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
)


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=0, help="Maximum number of cycles (0 = unlimited)")
@click.option("--resume", is_flag=True, help="Resume from saved state")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama", "gemini"]), default=None, help="LLM provider")
def start(goal: str | None, model: str | None, max_cycles: int, resume: bool, provider: str | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        from furrow.config import settings
        settings.model = model
    if provider:
        from furrow.config import Provider, settings
        settings.provider = Provider(provider)
    try:
        asyncio.run(Orchestrator(goal=goal, max_cycles=max_cycles, resume=resume).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def web(host: str, port: int) -> None:
    from furrow.web.server import run
    run(host=host, port=port)


if __name__ == "__main__":
    main()
