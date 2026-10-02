from __future__ import annotations

import asyncio
import sys

import click
import structlog
from rich.console import Console

from furrow import configure_logging
from furrow.config import settings
from furrow.core.orchestrator import Orchestrator

console = Console()
logger = structlog.get_logger(__name__)


@click.group()
def main() -> None:
    """Furrow: autonomous coding agent with infinite parallel development loop."""
    configure_logging(settings.log_level)


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--cycles", default=None, type=int, help="Maximum number of cycles (0 = infinite)")
@click.option("--planner-model", default=None, help="Override planner model")
@click.option("--worker-model", default=None, help="Override worker model")
@click.option("--tester-model", default=None, help="Override tester model")
@click.option("--provider", default=None, help="LLM provider (anthropic, openai, ollama)")
def start(
    goal: str | None,
    model: str | None,
    cycles: int | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    provider: str | None,
) -> None:
    """Start the Furrow development loop with the given goal."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    # Build a copy of settings so we don't mutate the global singleton
    config_overrides: dict = {}
    if model:
        config_overrides["model"] = model
    if cycles is not None:
        config_overrides["max_cycles"] = cycles
    if planner_model:
        config_overrides["planner_model"] = planner_model
    if worker_model:
        config_overrides["worker_model"] = worker_model
    if tester_model:
        config_overrides["tester_model"] = tester_model
    if provider:
        config_overrides["provider"] = provider

    config = settings.model_copy()
    for key, value in config_overrides.items():
        setattr(config, key, value)

    logger.info("Starting Furrow", goal=goal, config=config_overrides)

    try:
        asyncio.run(Orchestrator(goal=goal, settings=config).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def web(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the Furrow web UI server."""
    from furrow.web.server import run
    run(host=host, port=port)


if __name__ == "__main__":
    main()
