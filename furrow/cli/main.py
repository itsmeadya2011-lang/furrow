from __future__ import annotations

import asyncio
import sys
from typing import Any

import click
from rich.console import Console

from furrow.config import Provider, Settings, settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="LLM provider")
@click.option("--planner-model", default=None, help="Model for planner")
@click.option("--worker-model", default=None, help="Model for worker")
@click.option("--tester-model", default=None, help="Model for tester")
@click.option("--workspace", type=click.Path(path_type=Path), default=None, help="Workspace directory")
@click.option("--max-parallel", type=int, default=None, help="Max parallel tasks")
@click.option("--max-cycles", type=int, default=None, help="Max development cycles")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), default=None, help="Log level")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    workspace: Path | None,
    max_parallel: int | None,
    max_cycles: int | None,
    log_level: str | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    overrides: dict[str, Any] = {}
    if model:
        overrides["model"] = model
    if provider:
        overrides["provider"] = Provider(provider)
    if planner_model:
        overrides["planner_model"] = planner_model
    if worker_model:
        overrides["worker_model"] = worker_model
    if tester_model:
        overrides["tester_model"] = tester_model
    if workspace:
        overrides["workspace"] = workspace
    if max_parallel:
        overrides["max_parallel_tasks"] = max_parallel
    if max_cycles is not None:
        overrides["max_cycles"] = max_cycles
    if log_level:
        overrides["log_level"] = log_level

    if overrides:
        for key, value in overrides.items():
            setattr(settings, key, value)

    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--port", type=int, default=8000, help="Port to bind")
def web(host: str, port: int) -> None:
    from furrow.web.server import run
    run(host=host, port=port)


if __name__ == "__main__":
    main()
