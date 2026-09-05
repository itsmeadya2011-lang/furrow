from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import Provider, settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="furrow")
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--cycles", type=int, default=0, help="Max development cycles (0 = infinite)")
@click.option("--max-parallel", type=int, default=5, help="Max parallel tasks per cycle")
@click.option("--workspace", type=click.Path(path_type=Path), default=Path.cwd(), help="Workspace directory")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="LLM provider")
@click.option("--model", default=None, help="Override LLM model")
def start(goal: str | None, cycles: int, max_parallel: int, workspace: Path, provider: str | None, model: str | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    settings.max_cycles = cycles
    settings.max_parallel_tasks = max_parallel
    settings.workspace = workspace
    if provider:
        settings.provider = Provider(provider)
    if model:
        settings.model = model
    try:
        asyncio.run(Orchestrator(goal=goal, client=LLMClient()).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
