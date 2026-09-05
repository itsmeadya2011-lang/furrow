from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "ollama"]),
    default=None,
    help="LLM provider",
)
@click.option(
    "--workspace",
    type=click.Path(path_type=Path),
    default=None,
    help="Working directory for tests and file operations",
)
@click.option(
    "--max-cycles",
    type=int,
    default=None,
    help="Maximum number of planning cycles (0 = unlimited)",
)
@click.option(
    "--planner-model",
    default=None,
    help="Model for the planner agent",
)
@click.option(
    "--worker-model",
    default=None,
    help="Model for the worker agent",
)
@click.option(
    "--tester-model",
    default=None,
    help="Model for the tester agent",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Generate plan only, do not execute tasks",
)
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    workspace: Path | None,
    max_cycles: int | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    dry_run: bool,
) -> None:
    from furrow.config import Settings, get_settings

    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    s = get_settings()

    overrides: dict[str, Any] = {}
    if model:
        overrides["model"] = model
    if provider:
        from furrow.config import Provider
        overrides["provider"] = Provider(provider)
    if workspace:
        overrides["workspace"] = workspace
    if max_cycles is not None:
        overrides["max_cycles"] = max_cycles
    if planner_model:
        overrides["planner_model"] = planner_model
    if worker_model:
        overrides["worker_model"] = worker_model
    if tester_model:
        overrides["tester_model"] = tester_model

    if overrides:
        s = Settings(**{**s.model_dump(mode="json"), **overrides})

    try:
        if dry_run:
            from furrow.agents.planner import PlannerAgent
            from furrow.llm import LLMClient

            client = LLMClient(settings=s)
            plan = asyncio.run(PlannerAgent(client=client).plan(goal))
            console.print(Panel(Pretty(plan.model_dump()), title="Plan (dry run)", border_style="blue"))
            return

        asyncio.run(Orchestrator(goal=goal, client=LLMClient(settings=s)).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
def web() -> None:
    from furrow.web.server import run

    run()


@main.command()
def version() -> None:
    try:
        from importlib.metadata import version as pkg_version

        click.echo(pkg_version("furrow"))
    except Exception:
        click.echo("0.1.0")


if __name__ == "__main__":
    main()
