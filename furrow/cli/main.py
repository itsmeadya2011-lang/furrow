from __future__ import annotations

import asyncio
import signal
import sys

import click
from rich.console import Console

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    """Furrow: autonomous coding agent with an infinite parallel loop."""


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override default LLM model")
@click.option(
    "--max-cycles",
    default=None,
    type=int,
    help="Stop after this many cycles (0 = infinite).",
)
@click.option(
    "--provider",
    default=None,
    type=click.Choice(["anthropic", "openai", "ollama"]),
    help="LLM provider to use.",
)
def start(goal: str | None, model: str | None, max_cycles: int | None, provider: str | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if provider is not None:
        from furrow.config import Provider

        settings.provider = Provider(provider)

    orchestrator = Orchestrator(goal=goal, settings=settings)

    def _shutdown(_signum: int, _frame: object) -> None:
        console.print("\n[yellow]Stop requested. Finishing current cycle...[/yellow]")
        orchestrator.stop()
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run

    run()


if __name__ == "__main__":
    main()
