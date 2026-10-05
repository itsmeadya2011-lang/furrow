from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import Provider, settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override the LLM model.")
@click.option("--planner-model", default=None, help="Override the planner model.")
@click.option("--worker-model", default=None, help="Override the worker model.")
@click.option("--tester-model", default=None, help="Override the tester model.")
@click.option(
    "--provider",
    type=click.Choice([provider.value for provider in Provider]),
    default=None,
    help="LLM provider to use. Omit to keep the configured default.",
)
@click.option(
    "--max-cycles",
    type=int,
    default=None,
    help="Stop after N planning cycles. 0 means unlimited (the configured default).",
)
@click.option(
    "--max-parallel-tasks",
    type=int,
    default=None,
    help="Maximum tasks to run in parallel per cycle. Must be 1 or greater.",
)
def start(
    goal: str | None,
    model: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    provider: str | None,
    max_cycles: int | None,
    max_parallel_tasks: int | None,
) -> None:
    """Run Furrow until the goal is met (or a configured limit is reached)."""
    if max_cycles is not None and max_cycles < 0:
        raise click.BadParameter("must be 0 (unlimited) or greater", param_hint="--max-cycles")
    if max_parallel_tasks is not None and max_parallel_tasks < 1:
        raise click.BadParameter("must be 1 or greater", param_hint="--max-parallel-tasks")

    # Only overrides actually passed are applied; anything left as None keeps the
    # configured (env/.env/default) value.
    for name, value in (
        ("model", model),
        ("planner_model", planner_model),
        ("worker_model", worker_model),
        ("tester_model", tester_model),
    ):
        if value:
            setattr(settings, name, value)
    if provider:
        settings.provider = Provider(provider)
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if max_parallel_tasks is not None:
        settings.max_parallel_tasks = max_parallel_tasks

    if not goal or not goal.strip():
        try:
            goal = click.prompt("Enter your goal for Furrow")
        except (click.Abort, EOFError, KeyboardInterrupt):
            goal = None  # Ctrl-C or EOF at the prompt: leave cleanly, no traceback
    if not goal or not goal.strip():
        console.print("[yellow]No goal given. Nothing to do.[/yellow]")
        sys.exit(1)

    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run

    run()


if __name__ == "__main__":
    main()
