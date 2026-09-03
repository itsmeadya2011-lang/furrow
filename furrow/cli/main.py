from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

console = Console()


def _get_version() -> str:
    pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
    with open(pyproject) as f:
        for line in f:
            if line.startswith("version = "):
                return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("Version not found in pyproject.toml")


@click.group()
def main() -> None:
    """Autonomous coding agent with an infinite parallel development loop."""
    pass


@main.command()
@click.argument("goal", required=False, help="Development goal for Furrow to accomplish")
@click.option(
    "--model",
    default=None,
    help="Override the default LLM model for completions",
)
def start(goal: str | None, model: str | None) -> None:
    """Start an autonomous development loop."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        from furrow.config import settings

        settings.model = model
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    """Start the Furrow web interface."""
    from furrow.web.server import run

    run()


@main.command()
def status() -> None:
    """Display current Furrow configuration."""
    from furrow.config import settings

    console.print(f"[bold]Provider:[/bold] {settings.provider}")
    console.print(f"[bold]Model:[/bold] {settings.model}")
    console.print(f"[bold]Planner Model:[/bold] {settings.planner_model}")
    console.print(f"[bold]Worker Model:[/bold] {settings.worker_model}")
    console.print(f"[bold]Tester Model:[/bold] {settings.tester_model}")
    console.print(f"[bold]Ollama Base URL:[/bold] {settings.ollama_base_url}")
    console.print(f"[bold]Max Parallel Tasks:[/bold] {settings.max_parallel_tasks}")
    console.print(f"[bold]Max Cycles:[/bold] {settings.max_cycles}")
    console.print(f"[bold]Workspace:[/bold] {settings.workspace}")
    console.print(f"[bold]Log Level:[/bold] {settings.log_level}")


@main.command()
def stop() -> None:
    """Stop a running Furrow process."""
    console.print("[yellow]Furrow runs in the foreground. Use Ctrl+C to stop.[/yellow]")


@main.command()
def version() -> None:
    """Show the Furrow package version."""
    click.echo(_get_version())


if __name__ == "__main__":
    main()
