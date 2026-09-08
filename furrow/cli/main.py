from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from furrow.config import Provider, Settings, settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    """Furrow: Autonomous coding agent with infinite parallel development loop."""
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="LLM provider")
@click.option("--planner-model", default=None, help="Model for planner agent")
@click.option("--worker-model", default=None, help="Model for worker agent")
@click.option("--tester-model", default=None, help="Model for tester agent")
@click.option("--workspace", type=click.Path(path_type=Path), default=None, help="Workspace directory")
@click.option("--max-cycles", type=int, default=None, help="Max cycles (0 for unlimited)")
@click.option("--max-parallel-tasks", type=int, default=None, help="Max parallel tasks")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    workspace: Path | None,
    max_cycles: int | None,
    max_parallel_tasks: int | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if provider:
        settings.provider = Provider(provider)
    if planner_model:
        settings.planner_model = planner_model
    if worker_model:
        settings.worker_model = worker_model
    if tester_model:
        settings.tester_model = tester_model
    if workspace:
        settings.workspace = workspace.resolve()
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if max_parallel_tasks is not None:
        settings.max_parallel_tasks = max_parallel_tasks

    console.print(Panel.fit(
        f"[bold green]Furrow[/bold green]\n"
        f"Goal: {goal}\n"
        f"Provider: {settings.provider}\n"
        f"Model: {settings.model}\n"
        f"Workspace: {settings.workspace}",
        title="Furrow"
    ))

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
    console.print(f"[green]Starting Furrow web UI at http://{host}:{port}[/green]")
    run(host=host, port=port)


if __name__ == "__main__":
    main()
