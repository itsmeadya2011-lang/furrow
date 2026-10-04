from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import Provider, Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", default=None, help="Override LLM provider (anthropic, openai, ollama)")
@click.option("--max-cycles", type=int, default=0, help="Stop after N cycles (0 = unlimited)")
@click.option(
    "--workspace",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Workspace directory",
)
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int,
    workspace: Path | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    settings = Settings()
    if model:
        settings.model = model
    if provider:
        settings.provider = Provider(provider)
    if max_cycles:
        settings.max_cycles = max_cycles
    if workspace is not None:
        settings.workspace = Path(workspace)
    try:
        asyncio.run(Orchestrator(goal=goal, client=LLMClient(settings=settings)).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
