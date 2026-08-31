from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import Provider, Settings, settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option(
    "--provider",
    type=click.Choice([p.value for p in Provider]),
    default=None,
    help="Override LLM provider",
)
@click.option("--max-cycles", type=int, default=0, help="Max orchestrator cycles (0 = infinite)")
@click.option("--max-parallel", type=int, default=5, help="Max parallel worker tasks")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int,
    max_parallel: int,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    run_settings = Settings(
        provider=settings.provider,
        model=settings.model,
        planner_model=settings.planner_model,
        worker_model=settings.worker_model,
        tester_model=settings.tester_model,
        anthropic_api_key=settings.anthropic_api_key,
        openai_api_key=settings.openai_api_key,
        ollama_base_url=settings.ollama_base_url,
        max_parallel_tasks=settings.max_parallel_tasks,
        max_cycles=settings.max_cycles,
        workspace=settings.workspace,
        log_level=settings.log_level,
        request_timeout=settings.request_timeout,
        retry_attempts=settings.retry_attempts,
        state_file=settings.state_file,
    )
    if model:
        run_settings.model = model
    if provider:
        run_settings.provider = Provider(provider)
    run_settings.max_cycles = max_cycles
    run_settings.max_parallel_tasks = max_parallel

    client = LLMClient(settings=run_settings)
    try:
        asyncio.run(
            Orchestrator(
                goal=goal,
                client=client,
                max_cycles=max_cycles,
                max_parallel=max_parallel,
            ).run()
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
