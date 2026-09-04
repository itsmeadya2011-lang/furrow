from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import Provider, Settings
from furrow.core.orchestrator import Orchestrator

console = Console()


def _apply_settings(
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    max_parallel_tasks: int | None,
    workspace: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
) -> Settings:
    from furrow.config import settings

    if model:
        settings.model = model
    if provider:
        try:
            settings.provider = Provider(provider)
        except ValueError:
            raise click.BadParameter(f"Invalid provider: {provider}. Choose from: {[p.value for p in Provider]}")
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if max_parallel_tasks is not None:
        settings.max_parallel_tasks = max_parallel_tasks
    if workspace:
        settings.workspace = Path(workspace)
    if planner_model:
        settings.planner_model = planner_model
    if worker_model:
        settings.worker_model = worker_model
    if tester_model:
        settings.tester_model = tester_model
    return settings


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", default=None, help=f"LLM provider ({', '.join(p.value for p in Provider)})")
@click.option("--max-cycles", default=None, type=int, help="Max cycles (0 = unlimited)")
@click.option("--max-parallel-tasks", default=None, type=int, help="Max parallel tasks")
@click.option("--workspace", default=None, type=click.Path(path_type=str), help="Workspace directory")
@click.option("--planner-model", default=None, help="Override planner model")
@click.option("--worker-model", default=None, help="Override worker model")
@click.option("--tester-model", default=None, help="Override tester model")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    max_parallel_tasks: int | None,
    workspace: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    _apply_settings(model, provider, max_cycles, max_parallel_tasks, workspace, planner_model, worker_model, tester_model)
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
    try:
        from importlib.metadata import version as _version
        click.echo(_version("furrow"))
    except Exception:
        click.echo("0.1.0")


if __name__ == "__main__":
    main()
