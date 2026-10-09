from __future__ import annotations

import asyncio
import json
import sys

import click
from rich.console import Console

from furrow.config import Provider
from furrow.config import settings as global_settings
from furrow.core.orchestrator import Orchestrator
from furrow.core.state import clear_state, load_state

console = Console()


@click.group()
def main() -> None:
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
@click.option("--max-cycles", type=int, default=None, help="Halt after N cycles (0 = unlimited)")
@click.option("--max-parallel", type=int, default=None, help="Max parallel worker tasks per cycle")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
) -> None:
    """Start the Furrow development loop for a goal."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    overrides: dict = {}
    if model:
        overrides["model"] = model
    if provider:
        overrides["provider"] = provider
    if max_cycles is not None:
        overrides["max_cycles"] = max_cycles
    if max_parallel is not None:
        overrides["max_parallel_tasks"] = max_parallel
    # Build a per-run Settings copy so the global singleton is never mutated.
    run_settings = global_settings.model_copy(update=overrides)
    try:
        asyncio.run(Orchestrator(goal=goal, settings=run_settings).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def state() -> None:
    """Show the persisted state from the last Furrow run."""
    data = load_state()
    if data is None:
        console.print("[yellow]No saved state.[/yellow]")
        return
    console.print_json(json.dumps(data, indent=2))


@main.command()
def reset() -> None:
    """Clear the persisted Furrow state."""
    if clear_state():
        console.print("[green]State cleared.[/green]")
    else:
        console.print("[yellow]No saved state.[/yellow]")


@main.command()
def web() -> None:
    """Start the Furrow web UI."""
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()