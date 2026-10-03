from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Callable, Coroutine
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from furrow.config import Provider, settings

console = Console()

FALLBACK_VERSION = "0.1.0"
PROVIDER_CHOICES = [provider.value for provider in Provider]
MODEL_OVERRIDE_FIELDS = ("model", "planner_model", "worker_model", "tester_model", "log_level")
NUMERIC_OVERRIDE_FIELDS = ("max_cycles", "max_parallel_tasks")
SECRET_FIELDS = ("anthropic_api_key", "openai_api_key")


# --------------------------------------------------------------------------- options


def model_options(command: Callable[..., Any]) -> Callable[..., Any]:
    """Attach the shared provider/model/runtime overrides to a command.

    Every option defaults to ``None``/``False`` so an unset flag never changes
    behaviour: whatever the environment (env vars / ``.env``) configured wins.
    """
    options = [
        click.option(
            "--provider",
            type=click.Choice(PROVIDER_CHOICES),
            default=None,
            help="LLM provider to use",
        ),
        click.option("--model", default=None, help="Override the default LLM model"),
        click.option("--planner-model", default=None, help="Override the planner model"),
        click.option("--worker-model", default=None, help="Override the worker model"),
        click.option("--tester-model", default=None, help="Override the tester model"),
        click.option(
            "--max-cycles",
            type=int,
            default=None,
            help="Maximum number of development cycles (0 = unlimited)",
        ),
        click.option(
            "--max-parallel-tasks",
            type=int,
            default=None,
            help="Maximum number of tasks executed in parallel",
        ),
        click.option(
            "--workspace",
            "workspace",
            type=click.Path(file_okay=False, path_type=Path),
            default=None,
            help="Workspace directory the agent works in",
        ),
        click.option("--log-level", default=None, help="Logging level, e.g. DEBUG, INFO, WARNING"),
        click.option(
            "--debug",
            is_flag=True,
            default=False,
            help="Print the effective configuration and show full tracebacks",
        ),
    ]
    for option in reversed(options):
        command = option(command)
    return command


# --------------------------------------------------------------------------- helpers


def print_effective_settings() -> None:
    """Print the resolved configuration, with API keys redacted."""
    data = settings.model_dump()
    for field in SECRET_FIELDS:
        data[field] = "***" if data.get(field) else None
    console.print("[bold]Effective configuration:[/bold]")
    console.print_json(json.dumps(data, default=str))


def apply_overrides(overrides: dict[str, Any], debug: bool) -> None:
    """Write the CLI overrides onto the ``settings`` singleton.

    The orchestrator/agent ``LLMClient`` defaults to this very object, so
    mutating it here is what makes the flags take effect.
    """
    provider = overrides.get("provider")
    if provider:
        settings.provider = Provider(provider)
    for field in MODEL_OVERRIDE_FIELDS:
        value = overrides.get(field)
        if value:
            setattr(settings, field, value)
    for field in NUMERIC_OVERRIDE_FIELDS:
        value = overrides.get(field)
        if value is not None:
            # 0 is meaningful for max_cycles (unlimited), hence the explicit check.
            setattr(settings, field, value)
    workspace = overrides.get("workspace")
    if workspace is not None:
        settings.workspace = Path(workspace)
    log_level = overrides.get("log_level")
    if log_level:
        logging.getLogger("furrow").setLevel(log_level.upper())
    if debug:
        print_effective_settings()


def run_async(factory: Callable[[], Coroutine[Any, Any, Any]], debug: bool) -> Any:
    """Run an awaitable, turning failures into friendly CLI errors."""
    try:
        return asyncio.run(factory())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)
    except Exception as exc:
        if debug:
            raise
        console.print(f"[red]Error: {exc}[/red]")
        console.print("[dim]Re-run with --debug for the full traceback.[/dim]")
        sys.exit(1)


def resolve_goal(goal: str | None) -> str:
    if not goal:
        return str(click.prompt("Enter your goal for Furrow"))
    return goal


# --------------------------------------------------------------------------- commands


@click.group()
def main() -> None:
    """Furrow - an autonomous coding agent with an infinite development loop."""


@main.command()
@click.argument("goal", required=False)
@model_options
def start(goal: str | None, debug: bool, **overrides: Any) -> None:
    """Run the full development loop until the goal is met (or cycles run out)."""
    goal = resolve_goal(goal)
    apply_overrides(overrides, debug)

    from furrow.core.orchestrator import Orchestrator

    console.print(f"[bold green]Furrow[/bold green] working towards: {goal}")
    run_async(lambda: Orchestrator(goal=goal).run(), debug)


@main.command()
@click.argument("goal", required=False)
@model_options
def plan(goal: str | None, debug: bool, **overrides: Any) -> None:
    """Run only the planner and print the plan as JSON (dry run, no workers)."""
    goal = resolve_goal(goal)
    apply_overrides(overrides, debug)

    from furrow.agents.planner import PlannerAgent

    planner = PlannerAgent()
    result = run_async(lambda: planner.plan(goal), debug)
    print(result.model_dump_json(indent=2))


@main.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Host to bind the web UI to")
@click.option("--port", default=8000, show_default=True, type=int, help="Port to bind the web UI to")
def web(host: str, port: int) -> None:
    """Serve the web dashboard."""
    from furrow.web.server import run

    run(host=host, port=port)


@main.command()
def version() -> None:
    """Print the installed furrow version."""
    try:
        installed = _package_version("furrow")
    except PackageNotFoundError:
        installed = FALLBACK_VERSION
    console.print(f"furrow {installed}")


if __name__ == "__main__":
    main()