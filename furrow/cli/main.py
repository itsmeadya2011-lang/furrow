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
    """Furrow — autonomous coding agent with an infinite parallel development loop."""
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override the primary LLM model")
@click.option(
    "--max-cycles",
    default=None,
    type=int,
    help="Maximum number of development cycles (0 = unlimited)",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume a previously saved session instead of starting fresh",
)
@click.option(
    "--state-file",
    default=None,
    help="Path to the state file (default: .furrow/state.json)",
)
def start(
    goal: str | None,
    model: str | None,
    max_cycles: int | None,
    resume: bool,
    state_file: str | None,
) -> None:
    """Start the Furrow development loop with a goal."""
    from furrow.config import settings

    if model:
        settings.model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles

    if not goal:
        if resume:
            # Resume from existing state
            sm = StateManager(state_file=state_file)
            state = sm.load()
            if state is None:
                console.print("[red]No saved state found. Starting fresh.[/red]")
                goal = click.prompt("Enter your goal for Furrow")
            else:
                goal = state.original_goal
                console.print(f"[cyan]Resuming session: {goal}[/cyan]")
        else:
            goal = click.prompt("Enter your goal for Furrow")

    try:
        asyncio.run(
            Orchestrator(
                goal=goal,
                settings=settings if model or max_cycles else None,
                state_file=state_file,
            ).run()
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    """Start the Furrow web UI server."""
    from furrow.web.server import run

    run()


@main.command()
def status() -> None:
    """Show the current or most recent session status."""
    sm = StateManager()
    state = sm.load()
    if state is None:
        console.print("[yellow]No active session. Start one with: furrow start \"your goal\"[/yellow]")
        return

    console.print(f"\n[bold cyan]Session Status[/bold cyan]")
    console.print(f"  Goal: {state.original_goal}")
    console.print(f"  Current goal: {state.goal}")
    console.print(f"  Status: {state.status.value}")
    console.print(f"  Cycle: {state.cycle}")
    if state.tasks:
        console.print(f"\n[bold]Tasks:[/bold]")
        for task in state.tasks:
            status_color = {
                "completed": "green",
                "failed": "red",
                "pending": "yellow",
            }.get(task.status, "white")
            console.print(f"  [{status_color}]{task.id}[/{status_color}] {task.description} — {task.status}")
    if state.test_history:
        console.print(f"\n[bold]Last test result:[/bold]")
        last = state.test_history[-1]
        color = "green" if last.get("passed") else "red"
        console.print(f"  [{color}]passed={last.get('passed')}[/{color}] {last.get('summary', '')}")
    if state.errors:
        console.print(f"\n[bold red]Errors:[/bold]")
        for e in state.errors:
            console.print(f"  • {e}")


if __name__ == "__main__":
    main()
