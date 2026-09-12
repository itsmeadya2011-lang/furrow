from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import Provider, Settings
from furrow.core.orchestrator import Orchestrator

console = Console()


def _validate_nonnegative_int(ctx: click.Context, param: click.Parameter, value: int | None) -> int | None:
    if value is None:
        return None
    if value < 0:
        raise click.BadParameter(f"{value} is not a valid non-negative integer (must be >= 0).")
    return value


def _validate_positive_int(ctx: click.Context, param: click.Parameter, value: int | None) -> int | None:
    if value is None:
        return None
    if value <= 0:
        raise click.BadParameter(f"{value} is not a valid positive integer (must be > 0).")
    return value


def _resolve_workspace(ctx: click.Context, param: click.Parameter, value: Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).resolve()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="LLM provider")
@click.option("--model", default=None, help="Override default LLM model")
@click.option("--planner-model", default=None, help="Override planner model")
@click.option("--worker-model", default=None, help="Override worker model")
@click.option("--tester-model", default=None, help="Override tester model")
@click.option("--workspace", type=click.Path(path_type=Path), default=None, callback=_resolve_workspace, is_eager=True, help="Workspace directory")
@click.option("--max-cycles", type=int, default=None, callback=_validate_nonnegative_int, is_eager=True, help="Maximum number of cycles (0 = unlimited)")
@click.option("--max-parallel-tasks", type=int, default=None, callback=_validate_positive_int, is_eager=True, help="Maximum parallel tasks")
def start(
    goal: str | None,
    provider: str | None,
    model: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    workspace: Path | None,
    max_cycles: int | None,
    max_parallel_tasks: int | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    settings_kwargs = {}
    if provider:
        settings_kwargs["provider"] = Provider(provider)
    if model:
        settings_kwargs["model"] = model
    if planner_model:
        settings_kwargs["planner_model"] = planner_model
    if worker_model:
        settings_kwargs["worker_model"] = worker_model
    if tester_model:
        settings_kwargs["tester_model"] = tester_model
    if workspace:
        settings_kwargs["workspace"] = workspace
    if max_cycles is not None:
        settings_kwargs["max_cycles"] = max_cycles
    if max_parallel_tasks is not None:
        settings_kwargs["max_parallel_tasks"] = max_parallel_tasks

    settings = Settings(**settings_kwargs) if settings_kwargs else None

    try:
        asyncio.run(Orchestrator(goal=goal, settings=settings).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
