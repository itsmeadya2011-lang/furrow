from __future__ import annotations

import asyncio
import logging
import sys

import click
import structlog
from rich.console import Console

from furrow.config import Provider, settings
from furrow.core.orchestrator import Orchestrator

console = Console()

# Configure structlog logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(logging, settings.log_level.upper(), logging.INFO)
    ),
)


@click.group()
def main() -> None:
    """Furrow - Autonomous coding agent with infinite parallel development loop."""


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "ollama"]),
    default=None,
    help="Override LLM provider",
)
@click.option(
    "--max-cycles",
    type=int,
    default=None,
    help="Maximum development cycles (0 = unlimited)",
)
@click.option(
    "--parallel",
    type=int,
    default=None,
    help="Maximum parallel tasks per cycle",
)
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    parallel: int | None,
) -> None:
    """Start the infinite Furrow development loop with a GOAL."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    # Apply overrides to settings singleton
    if model:
        settings.model = model
    if provider:
        settings.provider = Provider(provider)
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if parallel is not None:
        settings.max_parallel_tasks = parallel

    try:
        asyncio.run(Orchestrator(goal=goal, settings=settings).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)
    except Exception as e:  # noqa: BLE001
        console.print(f"\n[red]Furrow encountered an error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind the web server")
@click.option("--port", type=int, default=8000, help="Port to bind the web server")
def web(host: str, port: int) -> None:
    """Start the Furrow web server."""
    from furrow.web.server import run

    run(host=host, port=port)


if __name__ == "__main__":
    main()
