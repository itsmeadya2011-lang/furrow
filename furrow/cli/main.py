from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.core.orchestrator import Orchestrator
from furrow.core.state import StateManager

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=None, type=int, help="Maximum cycles before stopping")
@click.option("--no-state", is_flag=True, help="Disable persistent state tracking")
def start(goal: str | None, model: str | None, max_cycles: int | None, no_state: bool) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        from furrow.config import settings
        settings.model = model
    if max_cycles is not None:
        from furrow.config import settings
        settings.max_cycles = max_cycles

    # Initialize state manager unless disabled
    state_manager = None
    if not no_state:
        state_manager = StateManager()
        session = state_manager.get_current_session()
        if session and session["status"] == "running":
            console.print(f"[yellow]Warning: Previous session {session['id']} was interrupted.[/yellow]")
            if click.confirm("Resume previous goal?", default=False):
                goal = session["goal"]
                console.print(f"[dim]Resuming goal: {goal}[/dim]")

    try:
        asyncio.run(Orchestrator(goal=goal, state_manager=state_manager).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


@main.command()
def history() -> None:
    """Show session history."""
    state_manager = StateManager()
    sessions = state_manager.get_all_sessions()
    if not sessions:
        console.print("[yellow]No session history found.[/yellow]")
        return

    console.print("[bold]Furrow Session History[/bold]\n")
    for session in sessions:
        status_color = {
            "completed": "green",
            "running": "yellow",
            "interrupted": "yellow",
            "error": "red",
        }.get(session["status"], "white")

        console.print(f"[bold]Session {session['id']}[/bold]")
        console.print(f"  Goal: {session['goal'][:80]}...")
        console.print(f"  Status: [{status_color}]{session['status']}[/{status_color}]")
        console.print(f"  Cycles: {session.get('cycles', 0)}")
        console.print(f"  Tasks completed: {len(session.get('tasks_completed', []))}")
        console.print(f"  Tasks failed: {len(session.get('tasks_failed', []))}")
        console.print(f"  Started: {session.get('started_at', 'unknown')}")
        if session.get("completed_at"):
            console.print(f"  Completed: {session['completed_at']}")
        console.print()


@main.command()
@click.confirmation_option(prompt="Are you sure you want to clear all history?")
def clear_history() -> None:
    """Clear all session history."""
    state_manager = StateManager()
    state_manager.clear_history()
    console.print("[green]Session history cleared.[/green]")


if __name__ == "__main__":
    main()
