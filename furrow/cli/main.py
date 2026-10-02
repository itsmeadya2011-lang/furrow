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
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "ollama"], case_sensitive=False),
    default=None,
    help="Override LLM provider",
)
@click.option("--planner-model", "planner_model", default=None, help="Override planner model")
@click.option("--worker-model", "worker_model", default=None, help="Override worker model")
@click.option("--tester-model", "tester_model", default=None, help="Override tester model")
@click.option(
    "--max-cycles",
    "max_cycles",
    default=0,
    type=int,
    help="Maximum number of cycles (0 = unlimited)",
)
@click.option(
    "--max-parallel",
    "max_parallel",
    default=5,
    type=int,
    help="Maximum parallel tasks",
)
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    max_cycles: int,
    max_parallel: int,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    from furrow.config import Provider, Settings
    from furrow.llm import LLMClient

    overrides: dict = {"max_cycles": max_cycles, "max_parallel_tasks": max_parallel}
    if model is not None:
        overrides["model"] = model
    if provider is not None:
        overrides["provider"] = Provider(provider)
    if planner_model is not None:
        overrides["planner_model"] = planner_model
    if worker_model is not None:
        overrides["worker_model"] = worker_model
    if tester_model is not None:
        overrides["tester_model"] = tester_model

    client = LLMClient(settings=Settings(**overrides))
    try:
        asyncio.run(Orchestrator(goal=goal, client=client).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
