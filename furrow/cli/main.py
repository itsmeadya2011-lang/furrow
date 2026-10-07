from __future__ import annotations

import asyncio
import sys
import logging

import click
from rich.console import Console

from furrow.config import Settings, Provider, settings
from furrow.core.orchestrator import Orchestrator

console = Console()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@click.group()
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), help="Set log level")
def main(log_level: str) -> None:
    _configure_logging(log_level)
    settings.log_level = log_level


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="LLM provider")
@click.option("--max-cycles", default=0, type=int, help="Maximum number of cycles (0 = unlimited)")
@click.option("--max-parallel", default=5, type=int, help="Maximum parallel tasks")
@click.option("--planner-model", default=None, help="Model for planning")
@click.option("--worker-model", default=None, help="Model for workers")
@click.option("--tester-model", default=None, help="Model for testing")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int,
    max_parallel: int,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    
    # Apply overrides
    if model:
        settings.model = model
    if provider:
        settings.provider = Provider(provider)
    if max_cycles > 0:
        settings.max_cycles = max_cycles
    if max_parallel > 0:
        settings.max_parallel_tasks = max_parallel
    if planner_model:
        settings.planner_model = planner_model
    if worker_model:
        settings.worker_model = worker_model
    if tester_model:
        settings.tester_model = tester_model
    
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--port", default=8000, type=int, help="Port to bind")
def web(host: str, port: int) -> None:
    from furrow.web.server import run
    run(host=host, port=port)


@main.command()
def version() -> None:
    from furrow import __version__
    console.print(f"Furrow version {__version__}")


if __name__ == "__main__":
    main()
