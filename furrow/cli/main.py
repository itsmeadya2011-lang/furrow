from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from furrow.config import Settings, settings
from furrow.core.orchestrator import Orchestrator, OrchestratorState

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=None, type=int, help="Maximum number of cycles")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="LLM provider")
def start(goal: str | None, model: str | None, max_cycles: int | None, provider: str | None) -> None:
    """Start a new Furrow development loop."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    if model:
        settings.model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if provider:
        settings.provider = provider

    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--state-file", default=None, help="Path to state file (default: .furrow_state.json)")
def resume(state_file: str | None) -> None:
    """Resume Furrow from a saved state file."""
    state_path = Path(state_file) if state_file else Path(".furrow_state.json")

    if not state_path.exists():
        console.print(f"[red]State file not found: {state_path}[/red]")
        sys.exit(1)

    try:
        state_data = state_path.read_text()
        state = OrchestratorState.from_json(state_data)
        console.print(f"[green]Resuming from cycle {len(state.cycles) + 1}[/green]")
        console.print(f"Goal: {state.goal}")

        asyncio.run(Orchestrator(goal=state.goal, state=state).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Failed to resume: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option("--state-file", default=None, help="Path to state file (default: .furrow_state.json)")
def status(state_file: str | None) -> None:
    """Show status of a Furrow run."""
    state_path = Path(state_file) if state_file else Path(".furrow_state.json")

    if not state_path.exists():
        console.print(f"[red]State file not found: {state_path}[/red]")
        sys.exit(1)

    try:
        state_data = state_path.read_text()
        state = OrchestratorState.from_json(state_data)

        console.print(f"[bold]Goal:[/bold] {state.goal}")
        console.print(f"[bold]Cycles:[/bold] {len(state.cycles)}")
        console.print(f"[bold]Tasks:[/bold] {len(state.all_tasks)} total, {len(state.completed_task_ids)} completed, {len(state.failed_task_ids)} failed")

        if state.cycles:
            table = Table(title="Cycle History")
            table.add_column("Cycle", style="cyan")
            table.add_column("Tasks", style="white")
            table.add_column("Completed", style="green")
            table.add_column("Failed", style="red")
            table.add_column("Tests", style="yellow")

            for cycle in state.cycles:
                completed = sum(1 for t in cycle.tasks if t.status == "completed")
                failed = sum(1 for t in cycle.tasks if t.status == "failed")
                test_status = "pass" if cycle.test_result and cycle.test_result.passed else "fail"
                table.add_row(
                    str(cycle.cycle_num),
                    str(len(cycle.tasks)),
                    str(completed),
                    str(failed),
                    test_status,
                )
            console.print(table)

    except Exception as e:
        console.print(f"[red]Failed to read status: {e}[/red]")
        sys.exit(1)


@main.command()
def config() -> None:
    """Show current configuration."""
    from rich.json import JSON

    config_dict = {
        "provider": settings.provider.value,
        "model": settings.model,
        "planner_model": settings.planner_model,
        "worker_model": settings.worker_model,
        "tester_model": settings.tester_model,
        "max_parallel_tasks": settings.max_parallel_tasks,
        "max_cycles": settings.max_cycles,
        "workspace": str(settings.workspace),
        "log_level": settings.log_level,
    }
    console.print(JSON.from_data(config_dict))


@main.command()
def web() -> None:
    """Start the Furrow web interface."""
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()