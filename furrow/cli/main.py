from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.option("--goal", default=None, help="Goal for Furrow")
@click.option("--model", default=None, help="Override default LLM model")
@click.option("--planner-model", default=None, help="Override planner LLM model")
@click.option("--worker-model", default=None, help="Override worker LLM model")
@click.option("--tester-model", default=None, help="Override tester LLM model")
@click.option("--max-cycles", default=None, type=int, help="Override max cycles")
@click.option(
    "--provider",
    default=None,
    type=click.Choice(["anthropic", "openai", "ollama"]),
    help="Override LLM provider",
)
@click.argument("goal_arg", metavar="GOAL", required=False)
def start(
    goal: str | None,
    model: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    max_cycles: int | None,
    provider: str | None,
    goal_arg: str | None,
) -> None:
    from furrow.config import settings

    chosen_goal = goal if goal else goal_arg
    if not chosen_goal:
        chosen_goal = click.prompt("Enter your goal for Furrow")

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
    if provider:
        settings.provider = provider

    try:
        asyncio.run(Orchestrator(goal=chosen_goal).run())
    except KeyboardInterrupt:
        try:
            console.file.flush()
        except Exception:
            pass
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--port", default=8000, type=int, help="Port to bind")
def web(host: str, port: int) -> None:
    from furrow.web.server import run

    run(host=host, port=port)


if __name__ == "__main__":
    main()
