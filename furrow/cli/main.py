from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import Provider, Settings, configure_logging, settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="Override provider")
@click.option("--max-cycles", default=None, type=int, help="Maximum number of cycles")
@click.option("--max-parallel", default=None, type=int, help="Maximum parallel tasks")
@click.option("--log-level", default=None, type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), help="Log level")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
    log_level: str | None,
) -> None:
    # Configure logging
    configure_logging(log_level or settings.log_level)

    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    # Build config with overrides
    provider_enum = Provider(provider) if provider else settings.provider
    config = Settings(
        provider=provider_enum,
        model=model or settings.model,
        planner_model=settings.planner_model,
        worker_model=settings.worker_model,
        tester_model=settings.tester_model,
        anthropic_api_key=settings.anthropic_api_key,
        openai_api_key=settings.openai_api_key,
        ollama_base_url=settings.ollama_base_url,
        max_parallel_tasks=max_parallel if max_parallel is not None else settings.max_parallel_tasks,
        max_cycles=max_cycles if max_cycles is not None else settings.max_cycles,
        workspace=settings.workspace,
        log_level=log_level or settings.log_level,
    )

    client = LLMClient(settings=config)
    try:
        asyncio.run(Orchestrator(goal=goal, client=client, config=config).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--log-level", default=None, type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), help="Log level")
def web(log_level: str | None) -> None:
    configure_logging(log_level or settings.log_level)
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
