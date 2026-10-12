from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import Provider, settings
from furrow.core.orchestrator import Orchestrator

console = Console()

PROVIDER_CHOICES = [p.value for p in Provider]


@click.group()
def main() -> None:
    """Furrow - autonomous coding agent with an infinite parallel development loop."""


def _apply_start_options(
    *,
    model: str | None,
    provider: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
    workspace: str | None,
) -> None:
    """Mutate the global settings object from CLI options (None means "not set")."""
    if provider:
        settings.provider = Provider(provider)
    if model:
        # A bare --model overrides every agent model so it actually takes effect.
        settings.model = model
        settings.planner_model = model
        settings.worker_model = model
        settings.tester_model = model
    if planner_model:
        settings.planner_model = planner_model
    if worker_model:
        settings.worker_model = worker_model
    if tester_model:
        settings.tester_model = tester_model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if max_parallel is not None:
        settings.max_parallel = max_parallel
    if workspace:
        settings.workspace = Path(workspace)


def _check_api_key() -> None:
    """Fail fast with a clear message when the selected provider lacks credentials."""
    provider = settings.provider
    if provider == Provider.ANTHROPIC:
        key = settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise click.ClickException(
                "ANTHROPIC_API_KEY is not set. Export it or set FURROW_ANTHROPIC_API_KEY."
            )
    elif provider == Provider.OPENAI:
        key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise click.ClickException(
                "OPENAI_API_KEY is not set. Export it or set FURROW_OPENAI_API_KEY."
            )
    elif provider == Provider.OLLAMA:
        if not settings.ollama_base_url:
            raise click.ClickException("Ollama base URL is empty; set FURROW_OLLAMA_BASE_URL.")


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override the model for every agent (planner/worker/tester).")
@click.option("--provider", type=click.Choice(PROVIDER_CHOICES), default=None, help="LLM provider.")
@click.option("--planner-model", default=None, help="Model for the planner agent.")
@click.option("--worker-model", default=None, help="Model for the worker agents.")
@click.option("--tester-model", default=None, help="Model for the tester agent.")
@click.option("--max-cycles", type=int, default=None, help="Stop after N cycles (0 = unlimited).")
@click.option("--max-parallel", type=int, default=None, help="Max tasks executed in parallel.")
@click.option("--workspace", type=click.Path(exists=True), default=None, help="Workspace directory to operate in.")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
    workspace: str | None,
) -> None:
    """Start the Furrow development loop for a goal."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    _apply_start_options(
        model=model, provider=provider, planner_model=planner_model,
        worker_model=worker_model, tester_model=tester_model, max_cycles=max_cycles,
        max_parallel=max_parallel, workspace=workspace,
    )
    _check_api_key()
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default=None, help="Bind host for the web UI.")
@click.option("--port", type=int, default=None, help="Bind port for the web UI.")
def web(host: str | None, port: int | None) -> None:
    """Launch the Furrow web UI."""
    from furrow.web.server import run

    kwargs: dict = {}
    if host:
        kwargs["host"] = host
    if port is not None:
        kwargs["port"] = port
    run(**kwargs)


if __name__ == "__main__":
    main()