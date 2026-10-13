from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import settings, setup_logging
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    setup_logging()


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--resume/--no-resume", default=True, help="Resume from saved state")
@click.option("--max-cycles", default=0, type=int, help="Maximum cycles to run (0 = unlimited)")
def start(goal: str | None, model: str | None, resume: bool, max_cycles: int) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if max_cycles > 0:
        settings.max_cycles = max_cycles
    try:
        asyncio.run(Orchestrator(goal=goal, resume=resume).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


@main.command()
def status() -> None:
    """Show current furrow state."""
    from furrow.config import State
    state = State.from_file(settings.state_file)
    if not state:
        console.print("[yellow]No saved state found.[/yellow]")
        return
    
    console.print(f"[bold]Goal:[/bold] {state.goal}")
    console.print(f"[bold]Cycles:[/bold] {state.current_cycle}/{state.cycles}")
    console.print(f"[bold]Tasks:[/bold] {len(state.tasks)} total")
    
    from furrow.config import TaskStatus
    for status in TaskStatus:
        tasks = [t for t in state.tasks if t.status == status]
        if tasks:
            console.print(f"  {status.value}: {len(tasks)}")
            for t in tasks:
                console.print(f"    - {t.id}: {t.description[:60]}")


@main.command()
def reset() -> None:
    """Reset saved state."""
    if settings.state_file.exists():
        settings.state_file.unlink()
        console.print(f"[green]Deleted {settings.state_file}[/green]")
    else:
        console.print("[yellow]No state file to delete.[/yellow]")


if __name__ == "__main__":
    main()
