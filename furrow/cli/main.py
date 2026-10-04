from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import Provider, settings
from furrow.core.orchestrator import Orchestrator

console = Console()

PROVIDER_CHOICES = [member.value for member in Provider]


def apply_setting_overrides(
    model: str | None = None,
    max_cycles: int | None = None,
    max_parallel_tasks: int | None = None,
    provider: str | None = None,
    workspace: str | None = None,
) -> None:
    """Apply CLI overrides to the global ``settings`` object.

    All overrides are applied here, in one place, *before* ``Orchestrator`` is
    constructed: the orchestrator builds its planner/worker/tester agents (which
    read the per-role model fields) inside ``__init__``.
    """
    if model:
        # ``--model`` must cover every per-role field, otherwise the agents keep
        # calling the provider with their own hard-coded defaults.
        settings.model = model
        settings.planner_model = model
        settings.worker_model = model
        settings.tester_model = model

    if max_cycles is not None:
        if max_cycles < 0:
            raise click.BadParameter(
                "must be >= 0 (0 means unlimited).", param_hint="--max-cycles"
            )
        settings.max_cycles = max_cycles

    if max_parallel_tasks is not None:
        if max_parallel_tasks < 1:
            raise click.BadParameter("must be >= 1.", param_hint="--max-parallel-tasks")
        settings.max_parallel_tasks = max_parallel_tasks

    if provider:
        try:
            settings.provider = Provider(provider)
        except ValueError as exc:
            valid = ", ".join(PROVIDER_CHOICES)
            raise click.BadParameter(
                f"unknown provider {provider!r} (choose from: {valid}).",
                param_hint="--provider",
            ) from exc

    if workspace:
        settings.workspace = Path(workspace).expanduser()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option(
    "--model",
    default=None,
    help="Override the LLM model for every role (planner, worker, tester).",
)
@click.option(
    "--max-cycles",
    type=int,
    default=None,
    help="Maximum number of plan/act/test cycles. 0 means unlimited.",
)
@click.option(
    "--max-parallel-tasks",
    type=int,
    default=None,
    help="Maximum number of tasks executed in parallel (>= 1).",
)
@click.option(
    "--provider",
    type=click.Choice(PROVIDER_CHOICES),
    default=None,
    help="LLM provider to use.",
)
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, dir_okay=True),
    default=None,
    help="Workspace directory the agents operate in.",
)
def start(
    goal: str | None,
    model: str | None,
    max_cycles: int | None,
    max_parallel_tasks: int | None,
    provider: str | None,
    workspace: str | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    apply_setting_overrides(
        model=model,
        max_cycles=max_cycles,
        max_parallel_tasks=max_parallel_tasks,
        provider=provider,
        workspace=workspace,
    )
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Host to bind to.")
@click.option("--port", type=int, default=8000, show_default=True, help="Port to bind to.")
def web(host: str, port: int) -> None:
    if not 0 <= port <= 65535:
        raise click.BadParameter("must be between 0 and 65535.", param_hint="--port")
    from furrow.web.server import run

    run(host=host, port=port)


if __name__ == "__main__":
    main()
