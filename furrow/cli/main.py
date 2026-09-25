from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import Provider, Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient
from furrow.logging import configure_logging

console = Console()


@click.group()
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), help="Set log level")
def main(log_level: str) -> None:
    configure_logging(log_level)


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", default=None, type=click.Choice(["anthropic", "openai", "ollama"]), help="LLM provider")
@click.option("--max-cycles", default=0, type=int, help="Maximum number of cycles (0 = unlimited)")
@click.option("--planner-model", default=None, help="Model for planner agent")
@click.option("--worker-model", default=None, help="Model for worker agent")
@click.option("--tester-model", default=None, help="Model for tester agent")
@click.option("--parallel", default=5, type=int, help="Max parallel tasks")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    parallel: int,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    settings = Settings()
    
    if model:
        settings.model = model
    if provider:
        settings.provider = Provider(provider)
    if max_cycles > 0:
        settings.max_cycles = max_cycles
    if planner_model:
        settings.planner_model = planner_model
    if worker_model:
        settings.worker_model = worker_model
    if tester_model:
        settings.tester_model = tester_model
    settings.max_parallel_tasks = parallel

    client = LLMClient(settings=settings)
    
    try:
        asyncio.run(Orchestrator(goal=goal, client=client, settings=settings).run())
        
        # Print cost summary
        cost = client.get_cost_summary()
        console.print("\n[bold cyan]Cost Summary:[/bold cyan]")
        console.print(f"  Total: ${cost['total_cost_usd']:.4f}")
        console.print(f"  Input tokens: {cost['total_input_tokens']}")
        console.print(f"  Output tokens: {cost['total_output_tokens']}")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
def web(host: str, port: int) -> None:
    from furrow.web.server import run
    run(host=host, port=port)


@main.command()
def version() -> None:
    """Show version information."""
    from furrow import __version__
    console.print(f"Furrow v{__version__}")


if __name__ == "__main__":
    main()