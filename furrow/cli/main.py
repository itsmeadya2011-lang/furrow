from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import Provider, settings
from furrow.core.orchestrator import Orchestrator
from furrow.logging import configure_logging

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", default=None, help="LLM provider: anthropic, openai, or ollama")
@click.option("--planner-model", default=None, help="Override planner model")
@click.option("--worker-model", default=None, help="Override worker model")
@click.option("--tester-model", default=None, help="Override tester model")
@click.option("--max-cycles", default=None, type=int, help="Maximum cycles (0 = infinite)")
@click.option("--max-parallel", default=None, type=int, help="Maximum parallel tasks per cycle")
@click.option("--workspace", default=None, type=click.Path(), help="Workspace directory")
@click.option("--log-level", default=None, help="Log level: DEBUG, INFO, WARNING, ERROR")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
    workspace: str | None,
    log_level: str | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    # Apply CLI overrides to the global settings
    if model:
        settings.model = model
    if provider:
        settings.provider = Provider(provider.lower())
    if planner_model:
        settings.planner_model = planner_model
    if worker_model:
        settings.worker_model = worker_model
    if tester_model:
        settings.tester_model = tester_model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if max_parallel is not None:
        settings.max_parallel_tasks = max_parallel
    if workspace:
        settings.workspace = Path(workspace)
    if log_level:
        configure_logging(log_level)
    else:
        configure_logging(settings.log_level)

    try:
        asyncio.run(Orchestrator(goal=goal, settings=settings).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
