from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import click
import structlog
from rich.console import Console

from furrow.config import Provider, Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

console = Console()


def _get_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return "0.1.0"
    try:
        return version("furrow")
    except PackageNotFoundError:
        return "0.1.0"


def _configure_logging(level: str) -> None:
    """Configure standard-library and structlog log levels."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level)
    try:
        structlog.configure(
            wrapper_class=structlog.make_filtering_logger(numeric_level),
        )
    except (AttributeError, TypeError):
        pass


def _validate_api_key(provider: Provider, settings: Settings) -> None:
    """Abort with a helpful message if the required API key is missing."""
    if provider == Provider.ANTHROPIC:
        key = settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            console.print("[red]Error: ANTHROPIC_API_KEY is not set.[/red]")
            console.print(
                "Set the FURROW_ANTHROPIC_API_KEY environment variable, "
                "or export ANTHROPIC_API_KEY before running."
            )
            sys.exit(1)
    elif provider == Provider.OPENAI:
        key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            console.print("[red]Error: OPENAI_API_KEY is not set.[/red]")
            console.print(
                "Set the FURROW_OPENAI_API_KEY environment variable, "
                "or export OPENAI_API_KEY before running."
            )
            sys.exit(1)


@click.group()
@click.version_option(version=_get_version(), prog_name="furrow")
def main() -> None:
    """Furrow: an autonomous coding agent with an infinite parallel development loop."""


@main.command()
@click.argument("goal", required=False)
@click.option(
    "--model",
    default=None,
    help="Override the default LLM model used across all agents.",
)
@click.option(
    "--workspace",
    type=click.Path(path_type=Path),
    default=None,
    help="Workspace directory to operate on (default: current directory).",
)
@click.option(
    "--provider",
    type=click.Choice([p.value for p in Provider], case_sensitive=False),
    default=None,
    help="LLM provider to use (default: anthropic).",
)
@click.option(
    "--max-cycles",
    type=int,
    default=None,
    help="Maximum number of planning cycles; 0 = unlimited (default: 0).",
)
@click.option(
    "--planner-model",
    default=None,
    help="LLM model to use for the planning agent.",
)
@click.option(
    "--worker-model",
    default=None,
    help="LLM model to use for the task-execution (worker) agent.",
)
@click.option(
    "--tester-model",
    default=None,
    help="LLM model to use for the testing agent.",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default=None,
    help="Logging verbosity (default: INFO).",
)
def start(
    goal: str | None,
    model: str | None,
    workspace: Path | None,
    provider: str | None,
    max_cycles: int | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    log_level: str | None,
) -> None:
    """Run Furrow autonomously against the given GOAL."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    overrides: dict[str, object] = {}
    if model:
        overrides["model"] = model
    if workspace:
        overrides["workspace"] = workspace
    if provider:
        overrides["provider"] = Provider(provider.lower())
    if max_cycles is not None:
        overrides["max_cycles"] = max_cycles
    if planner_model:
        overrides["planner_model"] = planner_model
    if worker_model:
        overrides["worker_model"] = worker_model
    if tester_model:
        overrides["tester_model"] = tester_model
    if log_level:
        overrides["log_level"] = log_level

    config = Settings(**overrides)
    _configure_logging(config.log_level)
    _validate_api_key(config.provider, config)

    try:
        client = LLMClient(settings=config)
        asyncio.run(Orchestrator(goal=goal, client=client, config=config).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host interface to bind the web server to (default: 0.0.0.0).",
)
@click.option(
    "--port",
    type=int,
    default=8000,
    help="TCP port to bind the web server to (default: 8000).",
)
def web(host: str, port: int) -> None:
    """Start the Furrow web (API + dashboard) server."""
    from furrow.web.server import run

    run(host=host, port=port)


if __name__ == "__main__":
    main()
