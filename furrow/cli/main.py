from __future__ import annotations

import asyncio
import sys

import click
import structlog
from rich.console import Console

from furrow.core.orchestrator import Orchestrator
from furrow.config import Provider, settings

logger = structlog.get_logger(__name__)
console = Console()


@click.group()
def main() -> None:
    """Furrow - Autonomous coding agent with infinite parallel development loop."""
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option(
    "--provider",
    type=click.Choice([p.value for p in Provider]),
    default=None,
    help="LLM provider (anthropic, openai, ollama)",
)
@click.option("--planner-model", default=None, help="Override planner model")
@click.option("--worker-model", default=None, help="Override worker model")
@click.option("--tester-model", default=None, help="Override tester model")
@click.option("--max-cycles", default=None, type=int, help="Maximum number of cycles (0 = unlimited)")
@click.option("--resume", is_flag=True, default=False, help="Resume from saved state file")
@click.option("--workspace", default=None, help="Workspace directory")
@click.option("--log-level", default=None, help="Log level (DEBUG, INFO, WARNING, ERROR)")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    max_cycles: int | None,
    resume: bool,
    workspace: str | None,
    log_level: str | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    # Apply overrides to settings.
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
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if workspace:
        settings.workspace = settings.workspace.__class__(workspace)
    if log_level:
        settings.log_level = log_level

    try:
        asyncio.run(
            Orchestrator(
                goal=goal,
                resume=resume,
            ).run()
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
def web(host: str, port: int) -> None:
    from furrow.web.server import run

    run(host=host, port=port)


if __name__ == "__main__":
    main()
