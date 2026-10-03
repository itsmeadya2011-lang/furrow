from __future__ import annotations

import asyncio
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import click
from rich.console import Console

from furrow.config import Provider, settings
from furrow.core.orchestrator import Orchestrator

console = Console()


def _resolve_version() -> str:
    """Installed version, or a placeholder when running from a source checkout."""
    try:
        return version("furrow")
    except PackageNotFoundError:
        return "0.0.0+source"


@click.group()
@click.version_option(version=_resolve_version(), prog_name="furrow")
def main() -> None:
    """Furrow: an autonomous coding agent that plans, builds and tests in a loop."""


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Model for every agent (overrides the per-agent ones).")
@click.option(
    "--provider",
    type=click.Choice([p.value for p in Provider], case_sensitive=False),
    default=settings.provider.value,
    show_default=True,
    help="LLM provider.",
)
@click.option(
    "--max-cycles",
    type=int,
    default=settings.max_cycles,
    show_default=True,
    help="Stop after this many cycles (0 = unlimited).",
)
@click.option(
    "--max-parallel-tasks",
    type=int,
    default=settings.max_parallel_tasks,
    show_default=True,
    help="Tasks to run concurrently in one wave.",
)
@click.option(
    "--max-idle-cycles",
    type=int,
    default=settings.max_idle_cycles,
    show_default=True,
    help="Stop after this many consecutive no-progress cycles (0 = unlimited).",
)
@click.option(
    "--workspace",
    type=click.Path(),
    default=str(settings.workspace),
    show_default=True,
    help="Directory the agents work in.",
)
@click.option(
    "--test-command",
    default=settings.test_command,
    help="Test command; auto-detected when omitted.",
)
def start(
    goal: str | None,
    model: str | None,
    provider: str,
    max_cycles: int,
    max_parallel_tasks: int,
    max_idle_cycles: int,
    workspace: str,
    test_command: str | None,
) -> None:
    """Run the plan/execute/test loop against GOAL (prompted for when omitted)."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    settings.provider = Provider(provider.lower())
    if model:
        # Every agent is handed its own model, so setting only ``model`` would
        # leave the planner, workers and tester on their configured defaults.
        settings.model = model
        settings.planner_model = model
        settings.worker_model = model
        settings.tester_model = model
    settings.max_cycles = max_cycles
    settings.max_parallel_tasks = max_parallel_tasks
    settings.max_idle_cycles = max_idle_cycles
    settings.workspace = Path(workspace)
    settings.test_command = test_command
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)
    except Exception as exc:
        console.print(f"\n[red]Furrow failed: {exc}[/red]")
        sys.exit(1)


@main.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Interface to bind.")
@click.option("--port", default=8000, show_default=True, help="Port to bind.")
def web(host: str, port: int) -> None:
    """Serve the web monitor, streaming loop output to the browser."""
    from furrow.web.server import run  # lazy: keeps FastAPI optional at CLI import time

    run(host=host, port=port)


if __name__ == "__main__":
    main()
