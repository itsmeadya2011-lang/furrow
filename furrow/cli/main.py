from __future__ import annotations

import asyncio
import logging
import sys

import click
import structlog
from rich.console import Console

from furrow.core.orchestrator import Orchestrator
from furrow.logging import logger

console = Console()


def _setup_logging(level: str = "INFO") -> None:
    """Configure structlog for console output."""
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


@click.group()
def main() -> None:
    """Furrow: autonomous coding agent with infinite parallel development loop."""
    _setup_logging()


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model for all agents")
def start(goal: str | None, model: str | None) -> None:
    """Start the Furrow infinite development loop with a goal."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    if model:
        # Override the default model for all agent roles
        from furrow.config import settings

        settings.model = model
        settings.planner_model = model
        settings.worker_model = model
        settings.tester_model = model
        logger.info("model_override", model=model)

    logger.info("furrow_start", goal=goal, model=model)
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        logger.info("furrow_stopped_by_user")
        sys.exit(0)


@main.command()
def web() -> None:
    """Start the Furrow web UI server."""
    from furrow.web.server import run

    _setup_logging()
    logger.info("furrow_web_start")
    run()


if __name__ == "__main__":
    main()
