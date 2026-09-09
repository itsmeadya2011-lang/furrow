from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "ollama"]),
    default=None,
    help="LLM provider",
)
@click.option("--max-cycles", type=int, default=None, help="Max planning cycles")
@click.option(
    "--workspace",
    type=click.Path(path_type=Path),
    default=None,
    help="Working directory",
)
@click.option("--planner-model", default=None, help="Planner model override")
@click.option("--worker-model", default=None, help="Worker model override")
@click.option("--tester-model", default=None, help="Tester model override")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    workspace: Path | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    from furrow.config import settings

    if model:
        settings.model = model
    if provider:
        from furrow.config import Provider

        settings.provider = Provider(provider)
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if workspace:
        settings.workspace = workspace
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
@click.option("--host", default="0.0.0.0", help="Bind address")
@click.option("--port", default=8000, type=int, help="Port number")
def web(host: str, port: int) -> None:
    from furrow.web.server import run

    run(host=host, port=port)


if __name__ == "__main__":
    main()
