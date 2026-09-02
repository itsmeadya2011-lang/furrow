from __future__ import annotations

import asyncio
import logging
import os
import sys

import click
import structlog
from rich.console import Console

from furrow.config import Provider, settings

console = Console()


def _setup_logging() -> structlog.stdlib.BoundLogger:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(sys.stderr),
    )
    return structlog.get_logger()


def _validate_provider_key() -> None:
    if settings.provider == Provider.ANTHROPIC and not settings.anthropic_api_key:
        console.print(
            "[red]Error: Provider 'anthropic' requires ANTHROPIC_API_KEY to be set.[/red]"
        )
        sys.exit(1)
    if settings.provider == Provider.OPENAI and not settings.openai_api_key:
        console.print(
            "[red]Error: Provider 'openai' requires OPENAI_API_KEY to be set.[/red]"
        )
        sys.exit(1)


def _apply_settings(
    provider: str | None = None,
    workspace: str | None = None,
    max_cycles: int | None = None,
    max_parallel_tasks: int | None = None,
    log_level: str | None = None,
    model: str | None = None,
) -> None:
    if provider:
        settings.provider = Provider(provider)
    if workspace:
        from pathlib import Path
        settings.workspace = Path(workspace).resolve()
    if max_cycles is not None:
        if max_cycles < 0:
            console.print("[red]Error: --max-cycles must be >= 0.[/red]")
            sys.exit(1)
        settings.max_cycles = max_cycles
    if max_parallel_tasks is not None:
        if max_parallel_tasks < 1:
            console.print("[red]Error: --max-parallel-tasks must be >= 1.[/red]")
            sys.exit(1)
        settings.max_parallel_tasks = max_parallel_tasks
    if log_level:
        settings.log_level = log_level
    if model:
        settings.model = model


@click.group()
@click.version_option(version="0.1.0", prog_name="furrow")
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "ollama"]),
    default=None,
    help="Choose provider (anthropic, openai, ollama)",
)
@click.option("--workspace", default=None, help="Set workspace directory")
@click.option(
    "--max-cycles",
    type=int,
    default=None,
    help="Set max cycles (0 = unlimited)",
)
@click.option(
    "--max-parallel-tasks",
    type=int,
    default=None,
    help="Set max parallel tasks",
)
@click.option("--log-level", default=None, help="Set log level")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    workspace: str | None,
    max_cycles: int | None,
    max_parallel_tasks: int | None,
    log_level: str | None,
) -> None:
    _apply_settings(
        provider=provider,
        workspace=workspace,
        max_cycles=max_cycles,
        max_parallel_tasks=max_parallel_tasks,
        log_level=log_level,
        model=model,
    )
    logger = _setup_logging()
    _validate_provider_key()
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        logger.error("Unexpected error", error=str(e))
        sys.exit(1)


@main.command()
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", type=int, default=8000, help="Server port")
def web(host: str, port: int) -> None:
    logger = _setup_logging()
    from furrow.web.server import run
    run(host=host, port=port)


if __name__ == "__main__":
    main()
