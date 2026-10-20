from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator
from furrow.core.state import load_state, restore_orchestrator, save_state, state_from_orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=None, type=int, help="Max cycles (0 = unlimited)")
@click.option("--max-parallel", default=None, type=int, help="Max parallel tasks")
@click.option("--state-file", default=None, type=click.Path(), help="Path to state file")
def start(
    goal: str | None,
    model: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
    state_file: str | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if max_parallel is not None:
        settings.max_parallel_tasks = max_parallel
    if state_file is not None:
        settings.state_file = Path(state_file)

    state_path = Path(settings.state_file)
    orch = Orchestrator(goal=goal)

    def on_event(event) -> None:
        try:
            if event.kind in ("cycle_end", "done"):
                try:
                    st = state_from_orchestrator(orch)
                    save_state(st, state_path)
                except Exception:
                    pass
        except Exception:
            pass

    orch.on_event = on_event
    try:
        asyncio.run(orch.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def status() -> None:
    state_path = Path(settings.state_file)
    try:
        state = load_state(state_path)
    except FileNotFoundError:
        console.print("[yellow]No saved run[/yellow]")
        return
    except Exception as exc:
        console.print(f"[red]Error loading state: {exc}[/red]")
        return

    tasks_by_status: dict[str, int] = {}
    for t in state.tasks:
        status_key = getattr(t.status, "value", str(t.status))
        tasks_by_status[status_key] = tasks_by_status.get(status_key, 0) + 1

    last_result = "n/a"
    if state.history:
        last = state.history[-1]
        passed = last.get("passed")
        summary = last.get("summary") or ""
        last_result = f"passed={passed}, summary={summary}"

    console.print("[bold]Saved run status[/bold]")
    console.print(f"Goal: {state.original_goal or state.goal}")
    console.print(f"Cycles run: {state.cycles}")
    console.print(f"Tasks: {len(state.tasks)} total ({tasks_by_status})")
    console.print(f"Last cycle: {last_result}")
    if state.updated_at:
        console.print(f"Updated at: {state.updated_at}")


@main.command()
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=None, type=int, help="Max cycles (0 = unlimited)")
@click.option("--max-parallel", default=None, type=int, help="Max parallel tasks")
@click.option("--state-file", default=None, type=click.Path(), help="Path to state file")
def resume(
    model: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
    state_file: str | None,
) -> None:
    if model:
        settings.model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if max_parallel is not None:
        settings.max_parallel_tasks = max_parallel
    if state_file is not None:
        settings.state_file = Path(state_file)

    state_path = Path(settings.state_file)
    try:
        state = load_state(state_path)
    except FileNotFoundError:
        console.print(
            f"[red]No saved run found at {state_path}; "
            f'start one with `furrow start "<goal>"`[/red]'
        )
        sys.exit(1)
    except Exception as exc:
        console.print(f"[red]Error loading state: {exc}[/red]")
        sys.exit(1)

    orch = Orchestrator(goal=state.goal)
    restore_orchestrator(orch, state)

    def on_event(event) -> None:
        try:
            if event.kind in ("cycle_end", "done"):
                try:
                    st = state_from_orchestrator(orch)
                    save_state(st, state_path)
                except Exception:
                    pass
        except Exception:
            pass

    orch.on_event = on_event
    try:
        asyncio.run(orch.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run

    run()


if __name__ == "__main__":
    main()
