from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.core.orchestrator import Orchestrator
from furrow.config import settings
from furrow.state import StateManager

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--resume", is_flag=True, help="Resume from saved state")
def start(goal: str | None, model: str | None, resume: bool) -> None:
    if not goal and not resume:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    try:
        asyncio.run(Orchestrator(goal=goal or "", model=model, resume=resume).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


@main.command()
def status() -> None:
    """Show progress summary from saved state."""
    state_manager = StateManager()
    state = state_manager.load()
    if not state:
        console.print("[yellow]No saved state found.[/yellow]")
        return
    summary = state.get_progress_summary()
    console.print("\n" + "=" * 50)
    console.print("[bold]Furrow Session Summary[/bold]")
    console.print("=" * 50)
    console.print(f"Goal: {summary['goal']}")
    console.print(f"Cycles: {summary['total_cycles']}")
    console.print(f"Tasks Planned: {summary['total_tasks_planned']}")
    console.print(f"Tasks Completed: {summary['total_tasks_completed']}")
    console.print(f"Tasks Failed: {summary['total_tasks_failed']}")
    console.print(f"Completion Rate: {summary['completion_rate']:.1%}")
    console.print(f"Created: {summary['created_at']}")
    console.print(f"Updated: {summary['updated_at']}")
    console.print("=" * 50)


@main.command()
@click.confirmation_option(prompt="Are you sure you want to clear the saved state?")
def clear() -> None:
    """Clear saved state."""
    state_manager = StateManager()
    state_manager.clear()
    console.print("[green]State cleared.[/green]")


if __name__ == "__main__":
    main()
