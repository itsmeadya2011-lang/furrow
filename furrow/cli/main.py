from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=0, type=int, help="Maximum number of cycles (0 = unlimited)")
@click.option("--state-file", default=None, type=click.Path(path_type=Path), help="State file for persistence")
@click.option("--provider", type=click.Choice(["anthropic", "openai"]), default=None, help="LLM provider")
@click.option("--parallel", default=5, type=int, help="Max parallel tasks")
def start(
    goal: str | None,
    model: str | None,
    max_cycles: int,
    state_file: Path | None,
    provider: str | None,
    parallel: int,
) -> None:
    """Start Furrow with a goal."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    settings = Settings()
    if model:
        settings.model = model
    if max_cycles > 0:
        settings.max_cycles = max_cycles
    if provider:
        settings.provider = provider  # type: ignore
    settings.max_parallel_tasks = parallel

    client = LLMClient(settings=settings)

    try:
        asyncio.run(Orchestrator(goal=goal, client=client, settings=settings, state_file=state_file).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
def web(host: str, port: int) -> None:
    """Start the web UI."""
    from furrow.web.server import run
    run(host=host, port=port)


@main.command()
@click.argument("goal")
@click.option("--model", default=None, help="Override LLM model")
def plan(goal: str, model: str | None) -> None:
    """Generate a plan for a goal without executing."""
    from furrow.agents.planner import PlannerAgent
    from furrow.llm import LLMClient

    settings = Settings()
    if model:
        settings.model = model

    client = LLMClient(settings=settings)
    planner = PlannerAgent(client=client, settings=settings)

    async def run_plan():
        plan = await planner.plan(goal)
        console.print(plan.model_dump_json(indent=2))

    asyncio.run(run_plan())


if __name__ == "__main__":
    main()