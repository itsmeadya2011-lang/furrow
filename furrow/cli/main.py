from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console

from furrow.config import Provider, settings as global_settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override all LLM model names")
@click.option(
    "--provider",
    type=click.Choice([provider.value for provider in Provider], case_sensitive=False),
    default=None,
    help="LLM provider",
)
@click.option(
    "--workspace",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Workspace directory",
)
@click.option(
    "--max-cycles",
    type=click.IntRange(min=0),
    default=None,
    help="Maximum planning cycles; 0 means unlimited",
)
@click.option("--test-command", default=None, help="Command used by the tester")
@click.option(
    "--test-timeout",
    type=click.FloatRange(min=0.0, min_open=True),
    default=None,
    help="Test command timeout in seconds",
)
@click.option(
    "--max-parallel-tasks",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum concurrent worker tasks",
)
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    workspace: Path | None,
    max_cycles: int | None,
    test_command: str | None,
    test_timeout: float | None,
    max_parallel_tasks: int | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    updates: dict[str, object] = {}
    if model:
        updates.update(
            {
                "model": model,
                "planner_model": model,
                "worker_model": model,
                "tester_model": model,
            }
        )
    if provider:
        updates["provider"] = Provider(provider.lower())
    if workspace is not None:
        updates["workspace"] = workspace
    if test_command is not None:
        updates["test_command"] = test_command
    if test_timeout is not None:
        updates["test_timeout"] = test_timeout
    if max_parallel_tasks is not None:
        updates["max_parallel_tasks"] = max_parallel_tasks

    run_settings = global_settings.model_copy(update=updates, validate=True)
    try:
        result = asyncio.run(
            Orchestrator(
                goal=goal,
                settings=run_settings,
                max_cycles=max_cycles,
            ).run()
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        raise click.exceptions.Exit(0)
    except Exception as exc:
        console.print(f"[red]Furrow failed: {exc}[/red]")
        raise click.exceptions.Exit(1)

    console.print(f"Run {result.status.value}: {result.message}")
    raise click.exceptions.Exit(result.exit_code)


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Host interface to bind")
@click.option(
    "--port",
    type=click.IntRange(1, 65535),
    default=8000,
    show_default=True,
    help="Port to bind",
)
def web(host: str, port: int) -> None:
    from furrow.web.server import run

    run(host=host, port=port)


if __name__ == "__main__":
    main()
