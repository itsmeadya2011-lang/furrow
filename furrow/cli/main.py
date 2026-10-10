from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator
from furrow.logging import setup_logging

console = Console()


@click.group()
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), help="Log level")
@click.option("--json-logs", is_flag=True, help="Output logs as JSON")
def main(log_level: str, json_logs: bool) -> None:
    settings.log_level = log_level
    settings.json_logs = json_logs
    setup_logging(level=log_level, json_logs=json_logs)


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="LLM provider")
@click.option("--max-cycles", default=10, help="Maximum number of cycles")
@click.option("--resume", is_flag=True, help="Resume from previous state")
def start(goal: str | None, model: str | None, provider: str | None, max_cycles: int, resume: bool) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if provider:
        settings.provider = provider
    try:
        asyncio.run(Orchestrator(goal=goal, max_cycles=max_cycles, resume=resume).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def web(host: str, port: int) -> None:
    from furrow.web.server import run
    run(host=host, port=port)


if __name__ == "__main__":
    main()
