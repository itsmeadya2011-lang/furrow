from __future__ import annotations

import asyncio
import json
import sys

import click
from rich.console import Console

from furrow.config import Provider, settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool) -> None:
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", type=click.Choice([p.value for p in Provider]), default=None, help="LLM provider")
@click.option("--max-cycles", type=int, default=None, help="Maximum number of cycles")
@click.option("--resume", is_flag=True, help="Resume from previous state file")
@click.option("--parallel", type=int, default=None, help="Max parallel tasks")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    resume: bool,
    parallel: int | None,
) -> None:
    if not goal and not resume:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if provider:
        settings.provider = Provider(provider)
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if parallel is not None:
        settings.max_parallel_tasks = parallel
    try:
        asyncio.run(Orchestrator(goal=goal or "", resume=resume).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def web(host: str, port: int) -> None:
    from furrow.web.server import run
    run(host=host, port=port)


@main.command()
def status() -> None:
    """Show status of current/previous Furrow run."""
    from furrow.config import OrchestratorState
    state_file = settings.state_file
    if state_file.exists():
        try:
            state = OrchestratorState.model_validate_json(state_file.read_text())
            console.print(f"[bold]Goal:[/bold] {state.goal}")
            console.print(f"[bold]Cycles:[/bold] {state.cycles}")
            console.print(f"[bold]Tasks:[/bold] {len(state.tasks)} total")
            for task in state.tasks:
                console.print(f"  {task.id}: {task.status} - {task.description}")
            if state.last_test_result:
                console.print(f"[bold]Last Test:[/bold] {'Passed' if state.last_test_result.passed else 'Failed'} - {state.last_test_result.summary}")
        except (json.JSONDecodeError, OSError, ValueError) as e:
            console.print(f"[red]Failed to read state: {e}[/red]")
    else:
        console.print("[yellow]No state file found.[/yellow]")


@main.command()
def clean() -> None:
    """Remove state file."""
    state_file = settings.state_file
    if state_file.exists():
        state_file.unlink()
        console.print("[green]State file removed.[/green]")
    else:
        console.print("[yellow]No state file to remove.[/yellow]")


if __name__ == "__main__":
    main()
