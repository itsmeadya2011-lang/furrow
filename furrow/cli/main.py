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
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="Set the LLM provider")
@click.option("--planner-model", default=None, help="Override planner model")
@click.option("--tester-model", default=None, help="Override tester model")
def start(goal: str | None, model: str | None, provider: str | None, planner_model: str | None, tester_model: str | None) -> None:
    from furrow.config import Provider, settings

    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if provider:
        settings.provider = Provider(provider)
    if planner_model:
        settings.planner_model = planner_model
    if tester_model:
        settings.tester_model = tester_model
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
