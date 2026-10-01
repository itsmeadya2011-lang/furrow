from __future__ import annotations

import asyncio
import logging
import sys

import click
import structlog
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

console = Console()


def configure_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stderr,
    )
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        stream=sys.stderr,
    )


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=None, type=int, help="Override max cycles (0 = unlimited)")
@click.option("--max-parallel", default=None, type=int, help="Override max parallel tasks")
@click.option("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
def start(
    goal: str | None,
    model: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
    log_level: str,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        from furrow.config import settings

        settings.model = model
    if max_cycles is not None:
        from furrow.config import settings

        settings.max_cycles = max_cycles
    if max_parallel is not None:
        from furrow.config import settings

        settings.max_parallel_tasks = max_parallel
    configure_logging(log_level)
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
def web(host: str, port: int) -> None:
    configure_logging("INFO")
    from furrow.web.server import run

    run(host=host, port=port)


if __name__ == "__main__":
    main()
