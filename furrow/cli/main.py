from __future__ import annotations

import asyncio
import sys

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
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="LLM provider")
@click.option("--max-cycles", default=None, type=int, help="Maximum number of cycles (0 = unlimited)")
@click.option("--max-parallel", default=None, type=int, help="Maximum parallel tasks")
@click.option("--workspace", default=None, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="Workspace directory")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
    workspace: str | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    
    settings = Settings()
    if model:
        settings.model = model
    if provider:
        settings.provider = provider
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if max_parallel is not None:
        settings.max_parallel_tasks = max_parallel
    if workspace:
        settings.workspace = workspace
    
    client = LLMClient(settings=settings)
    
    try:
        asyncio.run(Orchestrator(goal=goal, client=client, settings=settings).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
def web(host: str, port: int) -> None:
    from furrow.web.server import run
    run(host=host, port=port)


if __name__ == "__main__":
    main()