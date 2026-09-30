from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


def _apply_model_override(model: str | None) -> None:
    """When a custom model is supplied via CLI, update *all* model fields
    so the planner, worker, and tester use it consistently."""
    if model is None:
        return
    from furrow.config import settings
    settings.model = model
    settings.planner_model = model
    settings.worker_model = model
    settings.tester_model = model


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model for all stages")
@click.option("--max-cycles", default=None, type=int, help="Maximum number of dev cycles (0 = unlimited)")
def start(goal: str | None, model: str | None, max_cycles: int | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    _apply_model_override(model)
    try:
        orchestrator = Orchestrator(goal=goal)
        if max_cycles is not None:
            orchestrator.max_cycles = max_cycles
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--model", default=None, help="Override LLM model")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
def web(model: str | None, host: str, port: int) -> None:
    if model:
        _apply_model_override(model)
    from furrow.web.server import run
    run(host=host, port=port)


if __name__ == "__main__":
    main()
