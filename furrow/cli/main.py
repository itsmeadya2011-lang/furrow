from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import Settings, settings
from furrow.core.orchestrator import Orchestrator
from furrow.core.state import load_state, clear_state
from furrow.logging import configure_logging

console = Console()


@click.group()
def main() -> None:
    # Configure logging from settings
    configure_logging(settings.log_level)


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model for all agents")
@click.option("--planner-model", default=None, help="Override planner model")
@click.option("--worker-model", default=None, help="Override worker model")
@click.option("--tester-model", default=None, help="Override tester model")
@click.option("--max-cycles", default=None, type=int, help="Maximum number of cycles (0 = unlimited)")
@click.option("--max-parallel", default=None, type=int, help="Maximum parallel tasks per cycle")
@click.option("--resume/--no-resume", default=True, help="Resume from saved state")
@click.option("--clear-state", is_flag=True, help="Clear saved state before starting")
@click.option("--log-level", default=None, type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), help="Log level")
def start(
    goal: str | None,
    model: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
    resume: bool,
    clear_state: bool,
    log_level: str | None,
) -> None:
    if log_level:
        settings.log_level = log_level
        configure_logging(log_level)
    
    if clear_state:
        clear_state()
        console.print("[yellow]Cleared saved state.[/yellow]")
    
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    
    # Apply model overrides
    if model:
        settings.model = model
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
    
    # Try to resume from state
    orchestrator = None
    if resume:
        saved = load_state()
        if saved:
            console.print(f"[cyan]Resuming from cycle {saved.get('cycles', 0)}...[/cyan]")
            orchestrator = Orchestrator.from_state(saved)
            # Update goal if provided (allows changing direction)
            if goal != saved.get("goal"):
                orchestrator.goal = goal
    
    if orchestrator is None:
        orchestrator = Orchestrator(goal=goal)
    
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user. State saved.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
def web(host: str, port: int) -> None:
    from furrow.web.server import run
    run(host=host, port=port)


@main.command()
def clear() -> None:
    """Clear saved state."""
    clear_state()
    console.print("[green]State cleared.[/green]")


if __name__ == "__main__":
    main()
